import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _json_request(url: str, method: str, payload: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc


def _multipart_request(url: str, path: Path, base_url: str) -> dict[str, Any]:
    boundary = f"----broll-{uuid.uuid4().hex}"
    audio_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio_file\"; filename=\"{path.name}\"\r\nContent-Type: {audio_type}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    request = urllib.request.Request(
        url,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Idempotency-Key": path.name + "-" + uuid.uuid4().hex},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"API returned HTTP {exc.code}: {exc.read().decode(errors='replace')}") from exc


def submit(path: Path, base_url: str, output_dir: Path) -> None:
    endpoint = base_url.rstrip("/") + "/api/v1/broll/jobs"
    if path.suffix.lower() in {".mp3", ".wav", ".m4a"}:
        job = _multipart_request(endpoint, path, base_url)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        job = _json_request(endpoint, "POST", payload, {
            "Content-Type": "application/json",
            "Idempotency-Key": path.name + "-" + uuid.uuid4().hex,
        })
    else:
        payload = {"script": path.read_text(encoding="utf-8")}
        job = _json_request(endpoint, "POST", payload, {
            "Content-Type": "application/json",
            "Idempotency-Key": path.name + "-" + uuid.uuid4().hex,
        })

    job_id = job["job_id"]
    while True:
        status = _json_request(base_url.rstrip("/") + f"/api/v1/jobs/{job_id}", "GET")
        if status["status"] in {"succeeded", "failed"}:
            break
        time.sleep(1)
    job_dir = output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status_record = {"input_file": path.name, **status}
    (job_dir / "status.json").write_text(json.dumps(status_record, indent=2), encoding="utf-8")
    if status["status"] == "failed":
        print(f"{path.name}: failed ({status['error']['code']})")
        return
    for name in status["result"]["files"]:
        artifact_url = base_url.rstrip("/") + status["result"]["files"][name]
        with urllib.request.urlopen(artifact_url, timeout=120) as response:
            (job_dir / name).write_bytes(response.read())
    print(f"{path.name}: succeeded -> {job_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run script, JSON, and audio inputs through the B-roll API.")
    parser.add_argument("input", type=Path, help="An input file or a folder of .txt, .json, .mp3, .wav, and .m4a files")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("test-results"))
    args = parser.parse_args()
    paths = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.suffix.lower() in {".txt", ".json", ".mp3", ".wav", ".m4a"}:
            try:
                submit(path, args.base_url, args.output_dir)
            except Exception as exc:
                failure_dir = args.output_dir / f"failed-{path.stem}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                (failure_dir / "error.txt").write_text(str(exc), encoding="utf-8")
                print(f"{path.name}: {exc}")


if __name__ == "__main__":
    main()
