import time
import subprocess
import json
from dataclasses import replace
import uuid
from io import BytesIO

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import JobDatabase
from app.engine import _category
from app.main import create_app
from app.providers import search_pexels_clip
from app.storage import LocalStorage


def make_settings(tmp_path):
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        database_path=data_dir / "jobs.sqlite3",
        ffmpeg_path="ffmpeg",
        ffprobe_path="ffprobe",
        tts_provider="mock",
        tts_voice="en",
        visual_provider="procedural",
        require_stock_footage=True,
        pexels_api_key="",
        asr_provider="faster-whisper",
        asr_model="base",
        asr_device="cpu",
        storage_backend="local",
        s3_endpoint="localhost:9000",
        s3_bucket="test",
        s3_access_key="",
        s3_secret_key="",
        s3_secure=False,
        max_audio_seconds=600,
        max_upload_bytes=1024 * 1024,
        worker_count=1,
        min_scene_seconds=2.5,
    )


def test_pexels_is_default_visual_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("VISUAL_PROVIDER", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("DATABASE_PATH", raising=False)

    assert Settings.from_env().visual_provider == "pexels"
    assert Settings.from_env().require_stock_footage is True


def test_stock_required_job_fails_instead_of_returning_illustrations(monkeypatch, tmp_path):
    settings = replace(make_settings(tmp_path), visual_provider="pexels", pexels_api_key="test-key")
    monkeypatch.setattr("app.engine.search_pexels_clip", lambda *args: None)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "A neighborhood garden grows in the morning sun."},
            headers={"Idempotency-Key": "stock-required"},
        )
        assert response.status_code == 202
        completed = wait_for_status(client, response.json()["job_id"], "failed")
        assert completed["error"]["code"] == "STOCK_FOOTAGE_UNAVAILABLE"
        assert "sufficiently relevant Pexels footage" in completed["error"]["message"]


def wait_for_status(client, job_id, expected, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = client.get(f"/api/v1/jobs/{job_id}").json()
        if result["status"] in {"succeeded", "failed"}:
            assert result["status"] == expected, result
            return result
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_health_and_ready(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        health = client.get("/health", headers={"X-Correlation-Id": "trace-test"})
        assert health.json() == {"status": "ok"}
        assert health.headers["X-Correlation-Id"] == "trace-test"
        assert client.get("/ready").status_code == 200


def test_creator_ui_and_assets_are_served(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Start with your story" in page.text
        assert "/static/app.js" in page.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/styles.css").status_code == 200


def test_ambiguous_visual_terms_use_context():
    assert _category("Cloud computing powers online storage.", "") == "technology"
    assert _category("Clouds drift over the river.", "") == "nature"
    assert _category("The bank approved a loan.", "") == "finance"
    assert _category("A mouse crossed the field.", "") == "nature"
    assert _category("Python code processes data.", "") == "technology"
    assert _category("Trust and value are earned slowly.", "") == "abstract"


def test_script_job_is_idempotent_and_returns_artifacts(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    request_data = {
        "script": "Saving starts with a plan. Track your spending and make progress.",
        "aspect_ratios": ["16:9", "9:16"],
        "captions": {"burn_in": True, "srt": True},
    }
    with TestClient(app) as client:
        first = client.post("/api/v1/broll/jobs", json=request_data, headers={"Idempotency-Key": "job-1"})
        assert first.status_code == 202
        replay = client.post("/api/v1/broll/jobs", json=request_data, headers={"Idempotency-Key": "job-1"})
        assert replay.status_code == 202
        assert replay.json()["job_id"] == first.json()["job_id"]
        completed = wait_for_status(client, first.json()["job_id"], "succeeded")
        assert set(completed["result"]["files"]) == {
            "video_16_9.mp4", "video_9_16.mp4", "captions.srt", "timeline.json", "manifest.json"
        }
        video = client.get(completed["result"]["files"]["video_16_9.mp4"])
        assert video.status_code == 200
        assert video.content[4:8] == b"ftyp"
        portrait = client.get(completed["result"]["files"]["video_9_16.mp4"])
        assert portrait.status_code == 200
        landscape_path = settings.data_dir / "landscape.mp4"
        landscape_path.write_bytes(video.content)
        portrait_path = settings.data_dir / "portrait.mp4"
        portrait_path.write_bytes(portrait.content)
        import json

        for path, expected_size in ((landscape_path, (1920, 1080)), (portrait_path, (1080, 1920))):
            probe = subprocess.run(
                [
                    settings.ffprobe_path,
                    "-v", "error",
                    "-show_entries", "stream=width,height,r_frame_rate,codec_name",
                    "-of", "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            streams = json.loads(probe.stdout)["streams"]
            video_stream = next(stream for stream in streams if stream["codec_name"] == "h264")
            audio_stream = next(stream for stream in streams if stream["codec_name"] == "aac")
            assert (video_stream["width"], video_stream["height"], video_stream["r_frame_rate"]) == (*expected_size, "30/1")
            assert audio_stream["codec_name"] == "aac"
        frame_hashes = subprocess.run(
            [settings.ffmpeg_path, "-v", "error", "-i", str(landscape_path), "-an", "-f", "framemd5", "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        hashes = {
            line.rsplit(",", 1)[-1].strip()
            for line in frame_hashes.stdout.splitlines()
            if line and not line.startswith("#")
        }
        assert len(hashes) > 1
        timeline = client.get(completed["result"]["files"]["timeline.json"]).json()
        assert len(timeline["scenes"]) == 1
        assert timeline["scenes"][0]["start"] == 0
        assert timeline["scenes"][-1]["end"] == timeline["duration_seconds"]
        manifest = client.get(completed["result"]["files"]["manifest.json"]).json()
        assert manifest["cost_estimate_usd"] == 0.0
        assert manifest["rights"]["third_party_media"] is False


def test_idempotency_key_cannot_be_reused_for_different_payload(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        client.post("/api/v1/broll/jobs", json={"script": "One sentence."}, headers={"Idempotency-Key": "same"})
        response = client.post("/api/v1/broll/jobs", json={"script": "Different sentence."}, headers={"Idempotency-Key": "same"})
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"


def test_audio_upload_is_accepted_and_unsupported_extension_rejected(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            files={"audio_file": ("voice.wav", b"not-a-valid-wave-file", "audio/wav")},
            data={"script": "A narration."},
            headers={"Idempotency-Key": "upload"},
        )
        assert response.status_code == 202
        result = wait_for_status(client, response.json()["job_id"], "failed")
        assert result["error"]["code"] == "INVALID_AUDIO"
        wrong = client.post(
            "/api/v1/broll/jobs",
            files={"audio_file": ("voice.txt", b"not audio", "text/plain")},
            headers={"Idempotency-Key": "wrong-type"},
        )
        assert wrong.status_code == 415
        audio_only = client.post(
            "/api/v1/broll/jobs",
            files={"audio_file": ("voice.wav", b"not-a-valid-wave-file", "audio/wav")},
            headers={"Idempotency-Key": "audio-only-upload"},
        )
        assert audio_only.status_code == 202
        audio_only_result = wait_for_status(client, audio_only.json()["job_id"], "failed")
        assert audio_only_result["error"]["code"] == "INVALID_AUDIO"


def test_missing_key_and_empty_script_return_json_errors(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        missing_key = client.post("/api/v1/broll/jobs", json={"script": "A sentence."})
        assert missing_key.status_code == 400
        assert missing_key.json()["error"]["code"] == "IDEMPOTENCY_KEY_REQUIRED"
        invalid_json = client.post(
            "/api/v1/broll/jobs",
            content="{",
            headers={"Content-Type": "application/json", "Idempotency-Key": "bad-json"},
        )
        assert invalid_json.status_code == 400
        assert invalid_json.json()["error"]["code"] == "INVALID_JSON"
        empty = client.post(
            "/api/v1/broll/jobs",
            json={"script": " "},
            headers={"Idempotency-Key": "empty"},
        )
        assert empty.status_code == 422
        assert empty.json()["error"]["code"] == "INVALID_REQUEST"


def test_stock_provider_failure_falls_back_and_is_disclosed(tmp_path):
    settings = replace(make_settings(tmp_path), visual_provider="pexels", require_stock_footage=False)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "A plan helps you save money."},
            headers={"Idempotency-Key": "stock-fallback"},
        )
        completed = wait_for_status(client, response.json()["job_id"], "succeeded")
        timeline_url = completed["result"]["files"]["timeline.json"]
        manifest_url = completed["result"]["files"]["manifest.json"]
        timeline = client.get(timeline_url).json()
        manifest = client.get(manifest_url).json()
        assert timeline["scenes"][0]["clip"]["source"] == "procedural-illustration"
        assert manifest["warnings"][0].startswith("Pexels unavailable")


def test_pexels_candidates_are_ranked_by_title_overlap_and_never_reused(tmp_path, monkeypatch):
    settings = replace(make_settings(tmp_path), pexels_api_key="test-key")
    payload = {
        "videos": [
            {
                "id": 10,
                "url": "https://www.pexels.com/video/ocean-waves-123/",
                "video_files": [{"link": "https://videos.pexels.com/ocean.mp4", "file_type": "video/mp4"}],
            },
            {
                "id": 20,
                "url": "https://www.pexels.com/video/nurse-caring-for-patient-in-hospital-456/",
                "video_files": [{"link": "https://videos.pexels.com/hospital.mp4", "file_type": "video/mp4"}],
            },
        ]
    }
    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(payload).encode()),
    )
    monkeypatch.setattr("app.providers._download_video", lambda _url, path: path.write_bytes(b"video"))
    monkeypatch.setattr("app.providers._validate_video", lambda *_args: None)
    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.27)
    scene = {
        "text": "A nurse cares for a patient in hospital.",
        "visual_hint": "nurse caring for patient in hospital",
    }
    used_ids = set()

    chosen = search_pexels_clip(scene, used_ids, tmp_path / "chosen.mp4", settings)

    assert chosen is not None
    assert chosen["video_id"] == 20
    assert chosen["semantic_similarity"] == 0.27
    assert chosen["title_match_score"] >= 3
    assert chosen["visual_model"] == "OpenCLIP ViT-B/32"
    assert chosen["ai_generated"] is None
    assert chosen["commercial_use"].startswith("Permitted subject to the Pexels License")
    assert used_ids == {20}

    duplicate_only = {
        "videos": [payload["videos"][1]]
    }
    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(duplicate_only).encode()),
    )
    duplicate = search_pexels_clip(scene, used_ids, tmp_path / "duplicate.mp4", settings)
    assert duplicate is None

    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(payload).encode()),
    )
    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.19)
    low_score = search_pexels_clip(scene, set(), tmp_path / "low-score.mp4", settings)
    assert low_score is None

    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.27)
    no_match = search_pexels_clip(
        {"text": "A chef prepares fresh pasta.", "visual_hint": "chef cooking pasta"},
        used_ids,
        tmp_path / "none.mp4",
        settings,
    )
    assert no_match is None


def test_interrupted_job_is_requeued_when_service_starts(tmp_path):
    settings = make_settings(tmp_path)
    database = JobDatabase(settings.database_path)
    database.initialize()
    job_id = str(uuid.uuid4())
    database.create(
        job_id,
        "restart-recovery",
        "request-fingerprint",
        {"script": "This job continues after a service restart."},
        None,
        "correlation-recovery",
    )
    database.update(job_id, status="running", progress=45)
    app = create_app(settings, database, LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        completed = wait_for_status(client, job_id, "succeeded")
        assert completed["progress"] == 100
