import asyncio
import base64
import binascii
import hmac
import json
import os
from pathlib import Path
from typing import Any

import modal


APP_DIR = Path(__file__).parent
AUTH_SECRET_NAME = "broll-studio-browser-auth"
app = modal.App("broll-studio")
data_volume = modal.Volume.from_name("broll-studio-data", create_if_missing=True)
image = modal.Image.from_dockerfile(
    APP_DIR / "Dockerfile",
    context_dir=APP_DIR,
).env(
    {
        "DATA_DIR": "/service/data",
        "DATABASE_PATH": "/service/data/jobs.sqlite3",
        "STORAGE_BACKEND": "local",
        "JOB_EXECUTION_MODE": "inline",
        "WORKER_COUNT": "2",
        "TORCH_HOME": "/service/data/model-cache",
        "HF_HOME": "/service/data/model-cache/huggingface",
    }
)


class BrowserAuthentication:
    def __init__(self, asgi_app: Any) -> None:
        self.asgi_app = asgi_app
        self.username = os.environ["BROLL_STUDIO_USERNAME"].encode("utf-8")
        self.password = os.environ["BROLL_STUDIO_PASSWORD"].encode("utf-8")
        if not self.username or not self.password:
            raise RuntimeError("Broll Studio browser credentials must not be empty")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.asgi_app(scope, receive, send)
            return

        authorization = dict(scope.get("headers", [])).get(b"authorization", b"")
        authorized = False
        scheme, separator, encoded_credentials = authorization.partition(b" ")
        if separator and scheme.lower() == b"basic":
            try:
                decoded_credentials = base64.b64decode(encoded_credentials, validate=True).decode("utf-8")
                username, colon, password = decoded_credentials.partition(":")
                authorized = (
                    bool(colon)
                    and hmac.compare_digest(username.encode("utf-8"), self.username)
                    and hmac.compare_digest(password.encode("utf-8"), self.password)
                )
            except (binascii.Error, UnicodeDecodeError):
                authorized = False

        if authorized:
            await self.asgi_app(scope, receive, send)
            return

        body = b"Authentication required"
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"text/plain; charset=utf-8"),
                    (b"www-authenticate", b'Basic realm="Broll Studio", charset="UTF-8"'),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class CommitCompletedJobs:
    def __init__(self, asgi_app: Any, volume: modal.Volume) -> None:
        self.asgi_app = asgi_app
        self.volume = volume

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.asgi_app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")
        is_job_status = method == "GET" and path.startswith("/api/v1/jobs/") and "/artifacts/" not in path
        is_job_submission = method == "POST" and path == "/api/v1/broll/jobs"
        status_code = 500
        response_body = bytearray()

        async def capture_response(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            elif message["type"] == "http.response.body" and is_job_status:
                response_body.extend(message.get("body", b""))
            await send(message)

        await self.asgi_app(scope, receive, capture_response)

        should_commit_submission = is_job_submission and status_code == 202
        should_commit_terminal_status = False
        if is_job_status and status_code == 200:
            should_commit_terminal_status = json.loads(response_body).get("status") in {"succeeded", "failed"}

        if should_commit_submission or should_commit_terminal_status:
            await asyncio.to_thread(self.volume.commit)


@app.function(
    image=image,
    volumes={"/service/data": data_volume},
    cpu=4,
    memory=8192,
    secrets=[modal.Secret.from_name(AUTH_SECRET_NAME)],
    max_containers=1,
    scaledown_window=300,
    startup_timeout=900,
    timeout=3600,
)
@modal.concurrent(max_inputs=16)
@modal.asgi_app()
def serve() -> Any:
    from app.main import app as api_app

    return CommitCompletedJobs(BrowserAuthentication(api_app), data_volume)
