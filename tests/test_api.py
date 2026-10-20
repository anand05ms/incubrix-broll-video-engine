import time
import subprocess
import json
import sqlite3
import sys
import types
import hashlib
import hmac
from dataclasses import replace
from pathlib import Path
import uuid
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.callbacks import send_signed_callback
from app.db import JobDatabase
from app.engine import (
    EngineError,
    _category,
    _create_artwork,
    _fallback_visual_category,
    _is_visually_representable,
    _render_stock_clip,
    _render_video,
    _synthesize,
    _transcribe,
    _validate_rendered_video,
    _validate_rendered_segment,
    _validate_visual_asset,
)
from app.main import create_app
from app.providers import (
    StockProvidersExhausted,
    _call_provider_with_timeout,
    _cached_pixabay_search,
    search_internet_archive_clip,
    search_openverse_clip,
    search_pexels_clip,
    search_pixabay_clip,
    search_stock_clip,
    search_wikimedia_clip,
)
from app.scene_understanding import _offline_hint, enrich_scene_visual_hints
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
        visual_provider_order=("pixabay", "openverse", "wikimedia", "internet_archive", "pexels"),
        internet_archive_enabled=True,
        require_stock_footage=True,
        pexels_api_key="",
        pixabay_api_key="",
        openverse_client_id="",
        openverse_client_secret="",
        groq_api_key="",
        groq_model="openai/gpt-oss-120b",
        groq_vision_model="",
        asr_provider="faster-whisper",
        asr_model="base",
        asr_device="cpu",
        asr_compute_type="int8",
        asr_translate_for_search=True,
        storage_backend="local",
        s3_endpoint="localhost:9000",
        s3_bucket="test",
        s3_access_key="",
        s3_secret_key="",
        s3_secure=False,
        max_audio_seconds=600,
        max_upload_bytes=1024 * 1024,
        worker_count=1,
        job_execution_mode="inline",
        job_poll_interval=0.1,
        callback_hmac_secret="",
        min_scene_seconds=2.5,
    )


def test_wikimedia_is_default_stock_provider(monkeypatch, tmp_path):
    monkeypatch.delenv("VISUAL_PROVIDER", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.delenv("ASR_MODEL", raising=False)
    monkeypatch.delenv("ASR_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    monkeypatch.delenv("TTS_VOICE", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("DATABASE_PATH", raising=False)

    assert Settings.from_env().visual_provider == "auto"
    assert Settings.from_env().require_stock_footage is True
    assert Settings.from_env().asr_model == "small"
    assert Settings.from_env().asr_compute_type == "int8"
    assert Settings.from_env().groq_model == "openai/gpt-oss-120b"
    assert Settings.from_env().tts_provider == "kokoro"
    assert Settings.from_env().tts_voice == "af_heart"


def test_settings_load_dotenv_without_overriding_process_environment(monkeypatch, tmp_path):
    from dotenv import load_dotenv

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        f"PIXABAY_API_KEY=dotenv-test-key\nDATA_DIR={tmp_path / 'dotenv-data'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "app.config.load_dotenv",
        lambda override: load_dotenv(dotenv_path, override=override),
    )
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "shell-data"))

    settings = Settings.from_env()

    assert settings.pixabay_api_key == "dotenv-test-key"
    assert settings.data_dir == tmp_path / "shell-data"

    monkeypatch.setenv("PIXABAY_API_KEY", "process-test-key")
    assert Settings.from_env().pixabay_api_key == "process-test-key"


def test_scene_hints_use_offline_rules_without_groq_credentials(tmp_path):
    settings = make_settings(tmp_path)
    scenes = [
        {"text": "A nurse cares for a patient in a hospital.", "visual_hint": ""},
        {"text": "A gardener plants tomatoes.", "visual_hint": "close-up of tomato seedlings"},
    ]

    enriched, notice, provider = enrich_scene_visual_hints(scenes, settings)

    assert enriched[0]["visual_hint"] == "healthcare professionals caring for patients"
    assert enriched[1]["visual_hint"] == "close-up of tomato seedlings"
    assert notice and "offline scene rules" in notice
    assert provider == "offline-rules"
    assert scenes[0]["visual_hint"] == ""


def test_offline_search_hints_extract_concrete_visual_concepts():
    from app.providers import _search_query_variants

    examples = (
        (
            "Artificial intelligence is also changing the way people work.",
            "person using artificial intelligence on computer",
            "person artificial intelligence computer",
            "person analyzing data on computer",
        ),
        (
            "These tools help analyze large amounts of information.",
            "data analysis on computer screen",
            "data analysis computer screen",
            "person analyzing data on computer",
        ),
        (
            "AI can help students understand difficult concepts.",
            "person using artificial intelligence on computer",
            "person artificial intelligence computer",
            "person analyzing data on computer",
        ),
    )
    for narration, expected_hint, expected_first_query, expected_second_query in examples:
        hint = _offline_hint(narration)
        queries = _search_query_variants({"text": narration, "visual_hint": hint})
        assert hint == expected_hint
        assert queries[0] == expected_first_query
        assert queries[1] == expected_second_query
        assert narration.lower() not in queries


def test_team_planning_hint_avoids_fragmented_narration_query():
    from app.providers import _search_query_variants

    narration = "At the start of each day, teams plan their work and organize tasks."
    hint = _offline_hint(narration)
    queries = _search_query_variants({"text": narration, "visual_hint": hint})

    assert hint == "team planning meeting with task board"
    assert queries[0] == "team planning meeting task board"
    assert "each teams plan" not in queries
    assert len(queries) <= 2


def test_ui_explains_procedural_fallback_when_stock_is_unavailable():
    html = (Path(__file__).parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")

    assert "scene-specific illustration keeps the video complete" in html
    assert "the render fails rather than silently substituting" not in html
    assert "Real stock footage is required by default" not in html


def test_scene_hints_use_groq_for_missing_hints_and_preserve_explicit_hints(monkeypatch, tmp_path):
    settings = replace(make_settings(tmp_path), groq_api_key="test-groq-key")
    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"hints": [{"index": 1, "visual_hint": "hands harvesting ripe tomatoes"}]}
                    )
                }
            }
        ]
    }
    captured = {}

    def fake_urlopen(request, timeout):
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("app.scene_understanding.urllib.request.urlopen", fake_urlopen)
    scenes = [
        {"text": "Scene one.", "visual_hint": "explicit user hint"},
        {"text": "A gardener harvests tomatoes.", "visual_hint": ""},
    ]

    enriched, notice, provider = enrich_scene_visual_hints(scenes, settings)

    assert enriched[0]["visual_hint"] == "explicit user hint"
    assert enriched[1]["visual_hint"] == "hands harvesting ripe tomatoes"
    assert notice == "Narration text was sent to Groq for scene visual planning."
    assert provider == "groq"
    assert captured["authorization"] == "Bearer test-groq-key"
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["timeout"] == 20


def test_invalid_groq_scene_plan_falls_back_with_disclosure(monkeypatch, tmp_path):
    settings = replace(make_settings(tmp_path), groq_api_key="test-groq-key")
    monkeypatch.setattr(
        "app.scene_understanding.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(
            json.dumps({"choices": [{"message": {"content": '{"hints":[]}'}}]}).encode()
        ),
    )

    enriched, notice, provider = enrich_scene_visual_hints(
        [{"text": "A gardener tends a garden.", "visual_hint": ""}],
        settings,
    )

    assert enriched[0]["visual_hint"] == "people gardening and tending green plants"
    assert notice and "Groq scene planning failed" in notice
    assert provider == "offline-rules"


def test_stock_search_exhaustion_uses_scene_illustration_and_completes_job(monkeypatch, tmp_path):
    settings = replace(make_settings(tmp_path), visual_provider="pexels", pexels_api_key="test-key")

    def providers_exhausted(*_args):
        raise StockProvidersExhausted(
            [
                {
                    "provider": "pixabay",
                    "status": "skipped",
                    "reason": "required credentials are not configured (PIXABAY_API_KEY)",
                },
                {
                    "provider": "wikimedia",
                    "status": "no_relevant_clip",
                    "reason": "no licensed clip passed semantic relevance and media verification",
                },
                {
                    "provider": "openverse",
                    "status": "unavailable",
                    "reason": "Openverse request timed out",
                },
            ]
        )

    monkeypatch.setattr("app.engine.search_stock_clip", providers_exhausted)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "A neighborhood garden grows in the morning sun."},
            headers={"Idempotency-Key": "stock-required"},
        )
        assert response.status_code == 202
        completed = wait_for_status(client, response.json()["job_id"], "succeeded")
        timeline = client.get(completed["result"]["files"]["timeline.json"]).json()
        manifest = client.get(completed["result"]["files"]["manifest.json"]).json()
        captions = client.get(completed["result"]["files"]["captions.srt"]).text
        fallback_image = (
            settings.data_dir
            / "work"
            / response.json()["job_id"]
            / "artwork"
            / "16_9"
            / "scene_000.png"
        )

    attempts = timeline["provider_attempts"][0]["attempts"]
    assert attempts[0]["provider"] == "pixabay"
    assert timeline["scenes"][0]["visual_status"] == "procedural_illustration"
    assert timeline["scenes"][0]["clip"]["source"] == "procedural-illustration"
    assert timeline["scenes"][0]["clip"]["fallback"] is True
    assert timeline["scenes"][0]["clip"]["visual_category"] == "nature"
    assert timeline["scenes"][0]["clip"]["relevance_score"] is None
    assert timeline["scenes"][0]["clip"]["fallback_reason"] == "no_verified_stock_within_search_budget"
    assert Path(settings.data_dir / "work" / response.json()["job_id"] / "video_16_9.mp4").is_file()
    assert timeline["scenes"][0]["visual_assets"]["16:9"]["kind"] == "procedural_fallback"
    assert timeline["scenes"][0]["visual_assets"]["16:9"]["path"] == "clip_16_9_000.mp4"
    assert timeline["scenes"][0]["visual_assets"]["16:9"]["source_path"] == "artwork/16_9/scene_000.png"
    assert timeline["scenes"][0]["visual_assets"]["16:9"]["duration"] == pytest.approx(
        timeline["scenes"][0]["end"] - timeline["scenes"][0]["start"],
        abs=0.001,
    )
    assert timeline["scenes"][0]["visual_assets"]["16:9"]["fallback_used"] is True
    assert manifest["models"]["visuals"]["provider"] == "procedural-illustration"
    assert manifest["rights"]["third_party_media"] is False
    from PIL import Image, ImageStat

    with Image.open(fallback_image) as image:
        assert len(image.resize((20, 20)).getcolors(400)) > 10
        assert ImageStat.Stat(image.convert("L").resize((20, 20))).mean[0] > 70
    segment = settings.data_dir / "work" / response.json()["job_id"] / "clip_16_9_000.mp4"
    segment_probe = subprocess.run(
        [
            settings.ffprobe_path,
            "-v", "error",
            "-show_entries", "stream=codec_name,codec_type,avg_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json",
            str(segment),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    segment_media = json.loads(segment_probe.stdout)
    assert segment.stat().st_size > 0
    assert any(
        stream.get("codec_type") == "video" and stream.get("codec_name") == "h264"
        for stream in segment_media["streams"]
    )
    assert float(segment_media["format"]["duration"]) == pytest.approx(
        timeline["scenes"][0]["end"] - timeline["scenes"][0]["start"],
        abs=0.15,
    )
    _validate_rendered_segment(
        segment,
        settings,
        timeline["scenes"][0]["end"] - timeline["scenes"][0]["start"],
        require_visible=True,
    )
    assert "A neighborhood garden grows in the morning sun." in captions
    assert timeline["scenes"][-1]["end"] == timeline["duration_seconds"]
    assert any(
        "No sufficiently relevant footage found; scene-specific illustration used." in warning
        for warning in manifest["warnings"]
    )
    assert any("PIXABAY_API_KEY" in warning for warning in manifest["warnings"])
    assert any("scene-category illustrations" in warning for warning in manifest["warnings"])


def test_render_validation_rejects_blank_fallbacks_and_black_output(tmp_path):
    from PIL import Image

    settings = make_settings(tmp_path)
    blank_image = tmp_path / "blank.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(blank_image)
    with pytest.raises(EngineError, match="effectively black, blank, or uniform"):
        _validate_visual_asset(blank_image, "procedural")
    transparent_image = tmp_path / "transparent.png"
    Image.new("RGBA", (64, 64), (255, 255, 255, 0)).save(transparent_image)
    with pytest.raises(EngineError, match="transparent pixels"):
        _validate_visual_asset(transparent_image, "procedural")
    uniform_image = tmp_path / "uniform.png"
    Image.new("RGB", (64, 64), (120, 160, 190)).save(uniform_image)
    with pytest.raises(EngineError, match="effectively black, blank, or uniform"):
        _validate_visual_asset(uniform_image, "procedural")

    black_video = tmp_path / "black.mp4"
    subprocess.run(
        [
            settings.ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "color=c=black:s=160x90:r=30",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", "1", "-vf", "drawbox=x=0:y=ih*0.85:w=iw:h=ih*0.05:color=white:t=fill",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(black_video),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    with pytest.raises(EngineError, match="frames are entirely black"):
        _validate_rendered_video(black_video, settings, 1)


def test_fallback_artwork_is_opaque_scene_composition_at_each_aspect_ratio(tmp_path):
    from PIL import Image, ImageStat

    scene = {
        "text": "People use technology and computers to work.",
        "visual_hint": "people working with computers and technology",
    }
    expected_sizes = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}
    for ratio, expected_size in expected_sizes.items():
        output = tmp_path / f"fallback_{ratio.replace(':', '_')}.png"
        _create_artwork(scene, 0, ratio, output, include_text=False)
        _validate_visual_asset(output, "procedural", ratio)
        with Image.open(output) as image:
            assert image.size == expected_size
            assert image.mode == "RGB"
            pixels = image.resize((40, 40)).getdata()
            assert len(set(pixels)) > 20
            assert ImageStat.Stat(image.convert("L").resize((32, 32))).mean[0] > 70


def test_renderer_rebuilds_invalid_scene_visual_as_h264_fallback(tmp_path):
    settings = make_settings(tmp_path)
    broken_visual = tmp_path / "broken.png"
    broken_visual.write_bytes(b"not an image")
    audio_path = tmp_path / "audio.wav"
    subprocess.run(
        [
            settings.ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", "1", "-c:a", "pcm_s16le", str(audio_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    captions_path = tmp_path / "captions.srt"
    captions_path.write_text("", encoding="utf-8")
    visual = {"kind": "procedural", "path": broken_visual, "fallback_used": True}
    output = tmp_path / "rendered.mp4"

    _render_video(
        [visual],
        [{"text": "Technology at work.", "start": 0.0, "end": 1.0}],
        audio_path,
        "16:9",
        output,
        captions_path,
        False,
        1.0,
        settings,
        tmp_path,
    )

    assert visual["kind"] == "procedural_fallback"
    assert visual["path"].is_file()
    assert output.is_file() and output.stat().st_size > 0
    _validate_rendered_video(output, settings, 1.0)
    _validate_rendered_segment(tmp_path / "clip_16_9_000.mp4", settings, 1.0, True)


def test_burned_captions_have_dark_backing_on_light_fallback(tmp_path):
    from PIL import Image, ImageDraw

    settings = make_settings(tmp_path)
    artwork = tmp_path / "light-fallback.png"
    image = Image.new("RGB", (1920, 1080))
    draw = ImageDraw.Draw(image)
    for y in range(1080):
        level = 244 - y // 270
        draw.line((0, y, 1919, y), fill=(level, level, level))
    draw.rectangle((250, 250, 1670, 700), outline=(232, 236, 238), width=8)
    image.save(artwork)

    audio_path = tmp_path / "audio.wav"
    subprocess.run(
        [
            settings.ffmpeg_path, "-y",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", "1", "-c:a", "pcm_s16le", str(audio_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    captions_path = tmp_path / "captions.srt"
    captions_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nReadable caption contrast\n",
        encoding="utf-8",
    )
    output = tmp_path / "captioned.mp4"
    _render_video(
        [{"kind": "procedural", "path": artwork, "fallback_used": True}],
        [{"text": "Readable caption contrast.", "start": 0.0, "end": 1.0}],
        audio_path,
        "16:9",
        output,
        captions_path,
        True,
        1.0,
        settings,
        tmp_path,
    )

    frame = subprocess.run(
        [
            settings.ffmpeg_path, "-v", "error", "-ss", "0.5", "-i", str(output),
            "-frames:v", "1", "-vf", "crop=iw:ih/4:0:ih*0.75,format=rgb24",
            "-f", "rawvideo", "-",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    ).stdout
    assert len(frame) == 1920 * 270 * 3
    assert sum(
        1
        for red, green, blue in zip(frame[0::3], frame[1::3], frame[2::3])
        if max(red, green, blue) < 190
    ) > 500


def test_explicit_wikimedia_provider_falls_back_to_internet_archive_and_completes_job(
    monkeypatch, tmp_path
):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="wikimedia",
        visual_provider_order=("pixabay", "openverse", "wikimedia", "internet_archive", "pexels"),
        require_stock_footage=True,
    )
    provider_calls = []

    def no_relevant_wikimedia(*_args):
        provider_calls.append("wikimedia")
        return None

    def relevant_archive(_scene, _used_ids, destination, configured):
        provider_calls.append("internet_archive")
        subprocess.run(
            [
                configured.ffmpeg_path, "-y", "-f", "lavfi", "-i",
                "testsrc=size=320x180:rate=25", "-t", "1", "-c:v", "libx264",
                "-pix_fmt", "yuv420p", str(destination),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return {
            "path": destination,
            "source": "Internet Archive",
            "source_url": "https://archive.org/details/test-garden",
            "license": "CC BY (https://creativecommons.org/licenses/by/4.0/)",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Test contributor",
            "video_id": "archive-garden-1",
            "semantic_similarity": 0.16,
            "title_match_score": 4,
        }

    monkeypatch.setattr("app.providers.search_wikimedia_clip", no_relevant_wikimedia)
    monkeypatch.setattr("app.providers.search_internet_archive_clip", relevant_archive)
    monkeypatch.setattr("app.engine.choose_crop_anchor", lambda *_args: (0.5, 0.5))
    rendered_stock_inputs = []
    render_stock_clip = _render_stock_clip

    def capture_stock_render(video_path, *args, **kwargs):
        rendered_stock_inputs.append(video_path)
        return render_stock_clip(video_path, *args, **kwargs)

    monkeypatch.setattr("app.engine._render_stock_clip", capture_stock_render)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "Neighbors plant flowers together in a community garden."},
            headers={"Idempotency-Key": "commons-to-archive"},
        )
        assert response.status_code == 202
        completed = wait_for_status(client, response.json()["job_id"], "succeeded")
        timeline = client.get(completed["result"]["files"]["timeline.json"]).json()
        manifest = client.get(completed["result"]["files"]["manifest.json"]).json()
        rendered_video = client.get(completed["result"]["files"]["video_16_9.mp4"]).content

    assert provider_calls == ["wikimedia", "internet_archive"]
    assert len(rendered_stock_inputs) == 1
    assert rendered_stock_inputs[0].is_file()
    assert rendered_video[4:8] == b"ftyp"
    attempts = timeline["provider_attempts"][0]["attempts"]
    assert [(attempt["provider"], attempt["status"]) for attempt in attempts if attempt["status"] != "skipped"] == [
        ("wikimedia", "no_relevant_clip"),
        ("internet_archive", "selected"),
    ]
    assert timeline["scenes"][0]["clip"]["source"] == "Internet Archive"
    assert timeline["scenes"][0]["clip"]["fallback"] is True
    assert timeline["scenes"][0]["clip"]["fallback_reason"] == "low_confidence_contextual_match"
    assert timeline["scenes"][0]["clip"]["relevance_score"] == 0.16
    assert "Selected internet_archive after wikimedia" in timeline["provider_attempts"][0]["fallback_decision"]
    assert timeline["provider_attempts"] == manifest["visual_provider_attempts"]
    fallback_selection = manifest["visual_fallback_selections"][0]
    assert fallback_selection["scene_id"] == "scene-000"
    assert fallback_selection["provider"] == "Internet Archive"
    assert fallback_selection["license"] == "CC BY (https://creativecommons.org/licenses/by/4.0/)"
    assert fallback_selection["attribution"] == "Test contributor"
    assert fallback_selection["fallback_reason"] == "low_confidence_contextual_match"
    assert fallback_selection["relevance_score"] == 0.16
    assert any("Selected internet_archive after wikimedia" in warning for warning in manifest["warnings"])
    assert any(warning.startswith("LOW_CONFIDENCE_BROLL_FALLBACK:") for warning in manifest["warnings"])


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


def test_sqlalchemy_reads_and_updates_existing_sqlite_job_schema(tmp_path):
    database_path = tmp_path / "data" / "jobs.sqlite3"
    database_path.parent.mkdir(parents=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                request_hash TEXT NOT NULL,
                request_json TEXT NOT NULL,
                upload_path TEXT,
                status TEXT NOT NULL,
                progress INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                correlation_id TEXT NOT NULL,
                result_json TEXT,
                error_json TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs VALUES (
                ?, ?, ?, ?, NULL, 'running', 40, ?, ?, ?, NULL, NULL
            )
            """,
            (
                "legacy-job",
                "legacy-key",
                "legacy-hash",
                '{"script":"Keep this existing job"}',
                "2026-10-19T00:00:00+00:00",
                "2026-10-19T00:00:00+00:00",
                "legacy-correlation",
            ),
        )

    database = JobDatabase(database_path)
    database.initialize()
    assert database.get("legacy-job")["status"] == "running"
    queued = database.requeue_interrupted()

    assert [job["id"] for job in queued] == ["legacy-job"]
    assert database.get("legacy-job")["status"] == "queued"
    assert json.loads(database.by_idempotency_key("legacy-key")["request_json"]) == {
        "script": "Keep this existing job"
    }


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


def test_fallback_visual_categories_vary_by_scene_meaning():
    examples = {
        "A team uses a smartphone to chat and share messages.": "communication",
        "A business serves customers from a storefront.": "business",
        "Students learn through books and lessons.": "learning",
        "A student studies online with a laptop and books at a desk.": "learning",
        "Plan tasks using a calendar and checklist.": "productivity",
        "AI analyzes data on a computer network.": "technology",
        "Growth brings progress and success.": "growth",
        "Organized improvement cards make work clearer.": "improvement",
        "Cluttered tasks create a problem.": "problem",
    }
    for text, expected in examples.items():
        assert _fallback_visual_category({"text": text, "visual_hint": ""}) == expected


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


def test_stock_provider_failure_falls_back_and_is_disclosed(monkeypatch, tmp_path):
    import wave

    settings = replace(
        make_settings(tmp_path),
        visual_provider="pexels",
        visual_provider_order=("pexels",),
        internet_archive_enabled=False,
        require_stock_footage=False,
        tts_provider="kokoro",
    )
    monkeypatch.setattr("app.engine._package_version", lambda package: "0.9.4" if package == "kokoro" else None)

    def synthesize_silent_audio(_text, output, *_args):
        with wave.open(str(output), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(22050)
            audio.writeframes(b"\x00\x00" * 22050 * 5)

    monkeypatch.setattr("app.engine._synthesize", synthesize_silent_audio)
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
        assert manifest["models"]["tts"]["version"] == "0.9.4"
        assert timeline["provider_attempts"][0]["attempts"][0]["status"] == "skipped"
        assert timeline["scenes"][0]["visual_status"] == "procedural_illustration"
        assert any("video continues with narration" in warning for warning in manifest["warnings"])


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
    assert low_score is not None
    assert low_score["semantic_similarity"] == 0.19

    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.27)
    no_match = search_pexels_clip(
        {"text": "A chef prepares fresh pasta.", "visual_hint": "chef cooking pasta"},
        used_ids,
        tmp_path / "none.mp4",
        settings,
    )
    assert no_match is None


def test_wikimedia_video_requires_reusable_license_and_semantic_match(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 123,
                    "title": "File:Sunflower growing in a garden.webm",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/example/sunflower.webm",
                            "mime": "video/webm",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                                "Artist": {"value": "Alex Example"},
                            },
                        }
                    ],
                },
                {
                    "pageid": 456,
                    "title": "File:Sunflower growing in a garden.mp4",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/example/restricted.mp4",
                            "mime": "video/mp4",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-NC 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-nc/4.0/"},
                            },
                        }
                    ],
                },
            ]
        }
    }
    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(payload).encode()),
    )
    monkeypatch.setattr("app.providers._download_video", lambda _url, path: path.write_bytes(b"video"))
    monkeypatch.setattr("app.providers._validate_video", lambda *_args: None)
    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.31)

    selected = search_wikimedia_clip(
        {"text": "Sunflowers grow in the garden.", "visual_hint": "sunflower in garden"},
        set(),
        tmp_path / "commons.mp4",
        settings,
    )

    assert selected is not None
    assert selected["source"] == "Wikimedia Commons"
    assert selected["photographer_name"] == "Alex Example"
    assert selected["license_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert selected["semantic_similarity"] == 0.31
    assert selected["video_id"] == "123"


def test_internet_archive_uses_verified_creative_commons_video(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    search_results = {
        "response": {
            "docs": [
                {
                    "identifier": "restricted-garden",
                    "title": "Community garden planting flowers",
                    "description": "Neighbors plant flowers.",
                },
                {
                    "identifier": "licensed-garden",
                    "title": "Community garden planting flowers",
                    "description": "Neighbors plant flowers.",
                },
            ]
        }
    }
    metadata = {
        "restricted-garden": {
            "result": {
                "licenseurl": "https://creativecommons.org/licenses/by-nc/4.0/",
                "title": "Community garden",
            }
        },
        "licensed-garden": {
            "result": {
                "licenseurl": "https://creativecommons.org/licenses/by-sa/4.0/",
                "title": "Community garden",
                "creator": "Garden contributor",
            }
        },
    }
    files = {
        "result": [
            {
                "name": "large.mp4",
                "source": "original",
                "format": "MPEG4",
                "size": str(101 * 1024 * 1024),
            },
            {
                "name": "licensed-garden.mp4",
                "source": "original",
                "format": "MPEG4",
                "size": "2048",
            },
        ]
    }
    searched_urls = []

    def fake_search_json(url, *_args, **_kwargs):
        searched_urls.append(url)
        if "advancedsearch.php" in url:
            return search_results
        if url.endswith("/metadata"):
            return metadata[url.split("/")[-2]]
        if url.endswith("/files"):
            return files
        raise AssertionError(f"Unexpected Internet Archive URL: {url}")

    verified_urls = []

    def fake_verify(video_url, _scene, destination, _settings, candidate_number):
        verified_urls.append(video_url)
        candidate = destination.with_name(f"archive-candidate-{candidate_number}.mp4")
        candidate.write_bytes(b"verified video")
        return 0.31, candidate, None

    monkeypatch.setattr("app.providers._search_json", fake_search_json)
    monkeypatch.setattr("app.providers._verify_stock_file", fake_verify)
    used_ids = set()
    selected = search_internet_archive_clip(
        {
            "text": "Neighbors plant flowers together in a community garden.",
            "visual_hint": "community garden flowers growing",
        },
        used_ids,
        tmp_path / "archive.mp4",
        settings,
    )

    assert selected is not None
    assert selected["source"] == "Internet Archive"
    assert selected["video_id"] == "licensed-garden"
    assert selected["license"].startswith("CC BY-SA")
    assert selected["photographer_name"] == "Garden contributor"
    assert selected["semantic_similarity"] == 0.31
    assert used_ids == {"licensed-garden"}
    assert len([url for url in searched_urls if "advancedsearch.php" in url]) == 2
    assert verified_urls == [
        "https://archive.org/download/licensed-garden/licensed-garden.mp4"
    ]


def test_wikimedia_search_returns_no_candidates_for_unknown_or_noncommercial_license(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    payload = {
        "query": {
            "pages": [
                {
                    "pageid": 987,
                    "title": "File:Sunflower growing in a garden.mp4",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/example/restricted.mp4",
                            "mime": "video/mp4",
                            "extmetadata": {
                                "LicenseShortName": {"value": "CC BY-NC 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-nc/4.0/"},
                            },
                        }
                    ],
                }
            ]
        }
    }
    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: BytesIO(json.dumps(payload).encode()),
    )

    selected = search_wikimedia_clip(
        {"text": "Sunflowers grow in the garden.", "visual_hint": "sunflower in garden"},
        set(),
        tmp_path / "commons.mp4",
        settings,
    )

    assert selected is None


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


def test_database_atomically_claims_each_queued_job_once(tmp_path):
    database = JobDatabase(tmp_path / "jobs.sqlite3")
    database.initialize()
    job_id = str(uuid.uuid4())
    database.create(job_id, "queue-claim", "hash", {"script": "A queued job."}, None, "trace")

    claimed = database.claim_next()

    assert claimed is not None
    assert claimed["id"] == job_id
    assert claimed["status"] == "running"
    assert database.claim_next() is None


def test_external_execution_mode_leaves_new_job_for_separate_worker(tmp_path):
    settings = replace(make_settings(tmp_path), job_execution_mode="external")
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "A queued job for the worker process."},
            headers={"Idempotency-Key": "external-worker"},
        )

    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_pixabay_video_is_semantically_verified_and_attributed(tmp_path, monkeypatch):
    settings = replace(make_settings(tmp_path), pixabay_api_key="test-key")
    payload = {
        "hits": [
            {
                "id": 872,
                "pageURL": "https://pixabay.com/videos/nurse-caring-patient-hospital-872/",
                "tags": "nurse patient hospital",
                "user": "Example Creator",
                "videos": {"medium": {"url": "https://cdn.pixabay.com/video/nurse.mp4", "width": 1280, "height": 720}},
            }
        ]
    }
    monkeypatch.setattr("app.providers._cached_pixabay_search", lambda *_args: payload)

    def verify(_url, _scene, destination, _settings, _number):
        path = destination.with_name("candidate.mp4")
        path.write_bytes(b"verified")
        return 0.29, path, None

    monkeypatch.setattr("app.providers._verify_stock_file", verify)
    selected = search_pixabay_clip(
        {"text": "A nurse cares for a patient in hospital.", "visual_hint": "nurse caring patient hospital"},
        set(),
        tmp_path / "pixabay.mp4",
        settings,
    )

    assert selected is not None
    assert selected["source"] == "Pixabay"
    assert selected["photographer_name"] == "Example Creator"
    assert selected["semantic_similarity"] == 0.29
    assert selected["license_url"] == "https://pixabay.com/service/license-summary/"
    assert (tmp_path / "pixabay.mp4").read_bytes() == b"verified"


def test_pixabay_prefers_smallest_full_hd_variant_and_stops_after_confident_match(tmp_path, monkeypatch):
    settings = replace(make_settings(tmp_path), pixabay_api_key="test-key")
    payload = {
        "hits": [
            {
                "id": 872,
                "pageURL": "https://pixabay.com/videos/student-study-books-872/",
                "tags": "student laptop books study online learning",
                "user": "Example Creator",
                "videos": {
                    "large": {
                        "url": "https://cdn.pixabay.com/video/large.mp4",
                        "width": 3840,
                        "height": 2160,
                        "size": 13338271,
                    },
                    "medium": {
                        "url": "https://cdn.pixabay.com/video/medium.mp4",
                        "width": 2560,
                        "height": 1440,
                        "size": 6353902,
                    },
                    "small": {
                        "url": "https://cdn.pixabay.com/video/small.mp4",
                        "width": 1920,
                        "height": 1080,
                        "size": 4059830,
                    },
                    "tiny": {
                        "url": "https://cdn.pixabay.com/video/tiny.mp4",
                        "width": 1280,
                        "height": 720,
                        "size": 1588996,
                    },
                },
            },
            {
                "id": 873,
                "pageURL": "https://pixabay.com/videos/student-online-learning-873/",
                "tags": "student laptop books study online learning",
                "videos": {
                    "medium": {
                        "url": "https://cdn.pixabay.com/video/second-candidate.mp4",
                        "width": 1280,
                        "height": 720,
                    },
                },
            },
        ]
    }
    monkeypatch.setattr("app.providers._cached_pixabay_search", lambda *_args: payload)
    verified_urls = []

    def verify(url, _scene, destination, _settings, _number):
        verified_urls.append(url)
        path = destination.with_name("verified-candidate.mp4")
        path.write_bytes(b"verified")
        return 0.29, path, None

    monkeypatch.setattr("app.providers._verify_stock_file", verify)
    selected = search_pixabay_clip(
        {
            "text": "A student studies online with a laptop and books.",
            "visual_hint": "student learning with laptop and books",
        },
        set(),
        tmp_path / "pixabay.mp4",
        settings,
    )

    assert selected is not None
    assert verified_urls == ["https://cdn.pixabay.com/video/small.mp4"]
    assert selected["semantic_similarity"] == 0.29
    assert (tmp_path / "pixabay.mp4").read_bytes() == b"verified"


def test_pixabay_search_results_are_cached_for_twenty_four_hours(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    calls = []
    monkeypatch.setattr(
        "app.providers._search_json",
        lambda url: calls.append(url) or {"hits": [{"id": 1}]},
    )
    url = "https://pixabay.com/api/videos/?key=credential&q=garden"

    first = _cached_pixabay_search(url, settings)
    second = _cached_pixabay_search(url, settings)

    assert first == second == {"hits": [{"id": 1}]}
    assert len(calls) == 1


def test_openverse_filters_license_and_records_attribution(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        openverse_client_id="test-client",
        openverse_client_secret="test-secret",
    )
    responses = iter(
        [
            {
                "results": [
                    {
                        "id": "asset-1",
                        "title": "A nurse caring for a patient in hospital",
                        "url": "https://cdn.openverse.example/nurse.mp4",
                        "foreign_landing_url": "https://openverse.org/asset-1",
                        "license": "by-sa",
                        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                        "creator": "Example Nurse Footage",
                    },
                    {
                        "id": "asset-2",
                        "title": "Nurse patient hospital",
                        "url": "https://cdn.openverse.example/restricted.mp4",
                        "license": "by-nc",
                        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                    },
                ]
            },
        ]
    )
    search_headers = []

    def fake_search_json(_url, headers=None, data=None):
        if data is not None:
            return {"access_token": "test-token", "expires_in": 3600}
        search_headers.append(headers)
        return next(responses)

    monkeypatch.setattr("app.providers._search_json", fake_search_json)

    def verify(_url, _scene, destination, _settings, _number):
        path = destination.with_name("openverse-candidate.mp4")
        path.write_bytes(b"verified")
        return 0.32, path, None

    monkeypatch.setattr("app.providers._verify_stock_file", verify)
    selected = search_openverse_clip(
        {"text": "A nurse cares for a patient in hospital.", "visual_hint": "nurse patient hospital"},
        set(),
        tmp_path / "openverse.mp4",
        settings,
    )

    assert selected is not None
    assert selected["video_id"] == "asset-1"
    assert selected["photographer_name"] == "Example Nurse Footage"
    assert selected["license_url"] == "https://creativecommons.org/licenses/by-sa/4.0/"
    assert selected["semantic_similarity"] == 0.32
    assert search_headers[0]["Authorization"] == "Bearer test-token"


def test_openverse_token_uses_cached_client_credentials_form(tmp_path, monkeypatch):
    from urllib.parse import parse_qs
    from app.providers import _openverse_token

    settings = replace(
        make_settings(tmp_path),
        openverse_client_id="unique-test-client",
        openverse_client_secret="unique-test-secret",
    )
    requests = []

    def fake_search_json(url, headers=None, data=None):
        requests.append((url, headers, parse_qs(data.decode())))
        return {"access_token": "unique-test-token", "expires_in": 3600}

    monkeypatch.setattr("app.providers._OPENVERSE_TOKEN_CACHE", {})
    monkeypatch.setattr("app.providers._search_json", fake_search_json)

    assert _openverse_token(settings) == "unique-test-token"
    assert _openverse_token(settings) == "unique-test-token"
    assert len(requests) == 1
    url, headers, form = requests[0]
    assert url == "https://api.openverse.org/v1/auth_tokens/token/"
    assert headers == {"Content-Type": "application/x-www-form-urlencoded"}
    assert form == {
        "grant_type": ["client_credentials"],
        "client_id": ["unique-test-client"],
        "client_secret": ["unique-test-secret"],
    }


def test_openverse_rejected_credentials_have_safe_actionable_error(tmp_path, monkeypatch):
    from app.providers import StockProviderHTTPError, _openverse_token

    settings = replace(
        make_settings(tmp_path),
        openverse_client_id="unique-rejected-client",
        openverse_client_secret="unique-rejected-secret",
    )
    monkeypatch.setattr("app.providers._OPENVERSE_TOKEN_CACHE", {})
    monkeypatch.setattr(
        "app.providers._search_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StockProviderHTTPError("api.openverse.org", 401)
        ),
    )

    with pytest.raises(ValueError, match="Openverse rejected the configured client credentials") as error:
        _openverse_token(settings)

    assert "OPENVERSE_CLIENT_ID" in str(error.value)
    assert "unique-rejected-client" not in str(error.value)
    assert "unique-rejected-secret" not in str(error.value)


def test_stock_request_failure_logs_status_and_duration_without_query_or_secret(monkeypatch):
    import logging
    import json
    from io import StringIO
    import urllib.error

    from app.providers import StockProviderHTTPError, _search_json

    sensitive_query = "do-not-log-query"
    secret = "do-not-log-secret"
    provider_logger = logging.getLogger("broll")
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    from app.logging_config import JsonLogFormatter

    handler.setFormatter(JsonLogFormatter())
    provider_logger.addHandler(handler)
    url = f"https://api.openverse.org/v1/auth_tokens/token/?q={sensitive_query}&secret={secret}"
    http_error = urllib.error.HTTPError(url, 401, "Unauthorized", {}, BytesIO(b"{}"))
    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error),
    )

    try:
        with pytest.raises(StockProviderHTTPError) as error:
            _search_json(url)
    finally:
        provider_logger.removeHandler(handler)

    log_entry = json.loads(stream.getvalue())
    assert error.value.status_code == 401
    assert log_entry["provider_host"] == "api.openverse.org"
    assert log_entry["http_status"] == 401
    assert isinstance(log_entry["request_duration_ms"], int)
    assert sensitive_query not in stream.getvalue()
    assert secret not in stream.getvalue()


def test_stock_request_timeout_is_reported_explicitly(monkeypatch):
    import urllib.error

    from app.providers import _search_json

    monkeypatch.setattr(
        "app.providers.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError(TimeoutError())
        ),
    )

    with pytest.raises(ValueError, match="Stock provider request timed out"):
        _search_json("https://api.openverse.org/v1/videos/")


def test_openverse_requires_both_credentials(tmp_path):
    from app.providers import _openverse_token

    settings = replace(make_settings(tmp_path), openverse_client_id="only-client-id")

    with pytest.raises(ValueError, match="requires OPENVERSE_CLIENT_ID and OPENVERSE_CLIENT_SECRET"):
        _openverse_token(settings)


def test_kokoro_tts_uses_configured_voice_and_writes_pcm_audio(tmp_path, monkeypatch):
    import numpy as np

    captured = {}

    class FakePipeline:
        def __init__(self, lang_code):
            captured["lang_code"] = lang_code

        def __call__(self, text, voice):
            captured["text"] = text
            captured["voice"] = voice
            yield "text", "phonemes", np.array([0.0, 0.1], dtype=np.float32)

    def write(path, audio, sample_rate, subtype):
        captured["audio"] = audio
        captured["sample_rate"] = sample_rate
        captured["subtype"] = subtype
        Path(path).write_bytes(b"wav")

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=FakePipeline))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(write=write))
    settings = replace(make_settings(tmp_path), tts_provider="kokoro")
    output = tmp_path / "narration.wav"

    _synthesize("Kokoro narration test.", output, settings, "hf_alpha")

    assert output.read_bytes() == b"wav"
    assert captured["lang_code"] == "h"
    assert captured["voice"] == "hf_alpha"
    assert captured["sample_rate"] == 24000
    assert captured["subtype"] == "PCM_16"


def test_non_english_asr_keeps_caption_text_and_translates_search_hint(tmp_path, monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, *_args, **_kwargs):
            pass

        def transcribe(self, _path, *, task, **_kwargs):
            calls.append(task)
            segment = types.SimpleNamespace(
                text="नर्स मरीज की देखभाल करती है।" if task == "transcribe" else "A nurse cares for a patient.",
                start=0.0,
                end=2.0,
            )
            return [segment], types.SimpleNamespace(language="hi")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    transcript = _transcribe(
        tmp_path / "voice.wav",
        replace(make_settings(tmp_path), asr_translate_for_search=True),
    )

    assert transcript[0]["text"] == "नर्स मरीज की देखभाल करती है।"
    assert transcript[0]["search_text"] == "A nurse cares for a patient."
    assert calls == ["transcribe", "translate"]


def test_stock_render_uses_semantic_crop_anchor(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    commands = []
    monkeypatch.setattr("app.engine._run", lambda command, **_kwargs: commands.append(command))

    _render_stock_clip(
        tmp_path / "stock.mp4",
        tmp_path / "rendered.mp4",
        "9:16",
        4.0,
        settings,
        (0.75, 0.5),
    )

    video_filter = commands[0][commands[0].index("-vf") + 1]
    assert "crop=1080:1920:x='(iw-ow)*0.750':y='(ih-oh)*0.500'" in video_filter


def test_signed_callback_carries_verifiable_hmac_headers(monkeypatch):
    monkeypatch.setattr("app.callbacks.validate_callback_url", lambda _url: None)
    captured = {}

    class FakeResponse(BytesIO):
        status = 204

    def fake_open(request, timeout):
        captured["body"] = request.data
        captured["signature"] = request.get_header("X-incubrix-signature")
        captured["timestamp"] = request.get_header("X-incubrix-timestamp")
        captured["correlation_id"] = request.get_header("X-correlation-id")
        captured["timeout"] = timeout
        return FakeResponse(b"")

    monkeypatch.setattr("app.callbacks.urllib.request.build_opener", lambda *_args: types.SimpleNamespace(open=fake_open))

    send_signed_callback("https://hooks.example.com/jobs", "secret", {"status": "succeeded"}, "trace-42")

    expected = hmac.new(
        b"secret",
        captured["timestamp"].encode("ascii") + b"." + captured["body"],
        hashlib.sha256,
    ).hexdigest()
    assert captured["signature"] == f"sha256={expected}"
    assert captured["correlation_id"] == "trace-42"
    assert captured["timeout"] == 10


def test_callback_submission_requires_a_configured_hmac_secret(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    monkeypatch.setattr("app.main.validate_callback_url", lambda _url: None)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={"script": "A callback-enabled story.", "callback_url": "https://hooks.example.com/jobs"},
            headers={"Idempotency-Key": "callback-secret-required"},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "CALLBACK_NOT_CONFIGURED"


def test_auto_provider_falls_through_catalog_failure_to_commons(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        internet_archive_enabled=False,
        pixabay_api_key="configured-key",
    )
    attempts = []

    def pixabay(*_args):
        attempts.append("pixabay")
        raise ValueError("Pixabay temporarily unavailable")

    def commons(_scene, _used_ids, destination, _settings):
        attempts.append("wikimedia")
        destination.write_bytes(b"commons video")
        return {
            "path": destination,
            "source": "Wikimedia Commons",
            "video_id": "commons-1",
            "semantic_similarity": 0.3,
        }

    monkeypatch.setattr("app.providers.search_pixabay_clip", pixabay)
    monkeypatch.setattr("app.providers.search_wikimedia_clip", commons)
    selected = search_stock_clip(
        {"text": "A garden in spring.", "visual_hint": "flowers growing"},
        set(),
        tmp_path / "clip.mp4",
        settings,
    )

    assert selected["source"] == "Wikimedia Commons"
    assert selected["video_id"] == "commons-1"
    assert [(attempt["provider"], attempt["status"]) for attempt in selected["provider_attempts"] if attempt["status"] != "skipped"] == [
        ("pixabay", "unavailable"),
        ("wikimedia", "selected"),
    ]
    assert attempts == ["pixabay", "wikimedia"]


def test_internet_archive_is_appended_to_legacy_provider_order(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        visual_provider_order=("wikimedia",),
    )
    attempts = []

    def empty_commons(*_args):
        attempts.append("wikimedia")
        return None

    def archive_result(_scene, _used_ids, destination, _settings):
        attempts.append("internet_archive")
        destination.write_bytes(b"archive video")
        return {
            "path": destination,
            "source": "Internet Archive",
            "video_id": "licensed-archive-clip",
            "semantic_similarity": 0.3,
        }

    monkeypatch.setattr("app.providers.search_wikimedia_clip", empty_commons)
    monkeypatch.setattr("app.providers.search_internet_archive_clip", archive_result)

    selected = search_stock_clip(
        {"text": "A story about a night train crossing a mountain pass.", "visual_hint": ""},
        set(),
        tmp_path / "clip.mp4",
        settings,
    )

    assert selected["source"] == "Internet Archive"
    assert [attempt["provider"] for attempt in selected["provider_attempts"] if attempt["status"] != "skipped"] == [
        "wikimedia",
        "internet_archive",
    ]
    assert attempts == ["wikimedia", "internet_archive"]


def test_search_stock_clip_prefers_pixabay_and_stops_after_verified_clip(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        internet_archive_enabled=False,
        pixabay_api_key="configured-key",
        visual_provider_order=("pixabay", "wikimedia"),
    )
    observed_queries = []

    def candidate(source, video_id, score):
        def search(scene, used_ids, destination, _settings):
            observed_queries.append((source, scene["visual_hint"]))
            if video_id in used_ids:
                return None
            destination.write_bytes(source.encode())
            used_ids.add(video_id)
            return {
                "path": destination,
                "source": source,
                "video_id": video_id,
                "semantic_similarity": score,
            }

        return search

    monkeypatch.setattr(
        "app.providers.search_pixabay_clip",
        candidate("Pixabay", "pixabay-1", 0.29),
    )
    monkeypatch.setattr(
        "app.providers.search_wikimedia_clip",
        candidate("Wikimedia Commons", "commons-1", 0.34),
    )

    destination = tmp_path / "selected.mp4"
    used_video_ids = set()
    selected = search_stock_clip(
        {
            "text": "Small businesses use technology to improve customer service.",
            "visual_hint": "business owner helping customers",
        },
        used_video_ids,
        destination,
        settings,
    )

    assert selected["source"] == "Pixabay"
    assert selected["semantic_similarity"] == 0.29
    assert selected["fallback"] is False
    assert selected["relevance_score"] == 0.29
    assert destination.read_bytes() == b"Pixabay"
    assert used_video_ids == {"pixabay-1"}
    assert len(observed_queries) == 1
    assert any(
        attempt["provider"] == "pixabay" and attempt["status"] == "selected"
        for attempt in selected["provider_attempts"]
    )
    assert not list(tmp_path.glob("*.candidate.mp4"))


def test_search_query_variants_keep_visual_hint_and_expand_narration_terms():
    from app.providers import MAX_QUERIES_PER_SCENE, _search_query_variants

    variants = _search_query_variants(
        {
            "text": "Small businesses are increasingly using artificial intelligence to improve customer service.",
            "visual_hint": "business owner helping customers",
        }
    )

    assert variants[0] == "business owner helping customers"
    assert len(variants) == MAX_QUERIES_PER_SCENE

    abstract_variants = _search_query_variants(
        {
            "text": (
                "Organizing tasks at the beginning of the day can help reduce "
                "distractions and improve productivity."
            ),
            "visual_hint": "daily work planning",
        }
    )
    assert any("writing a daily task list" in variant for variant in abstract_variants)
    assert len(abstract_variants) <= MAX_QUERIES_PER_SCENE


def test_abstract_scene_keeps_its_own_fallback_when_later_scene_finds_stock(monkeypatch, tmp_path):
    settings = replace(make_settings(tmp_path), visual_provider="pixabay", pixabay_api_key="test-key")
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))

    stock_searches = []

    def stock_search(scene, _used_ids, destination, _settings):
        stock_searches.append(scene["_scene_id"])
        destination.write_bytes(b"verified stock media")
        return {
            "path": destination,
            "source": "Pixabay",
            "source_url": "https://pixabay.com/videos/example",
            "license": "Pixabay Content License",
            "attribution": "Pixabay contributor",
            "video_id": "clip-abstract-support",
            "semantic_similarity": 0.29,
            "relevance_score": 0.29,
            "fallback": False,
            "provider_attempts": [{"provider": "pixabay", "status": "selected", "reason": "verified"}],
            "search_duration_ms": 10,
        }

    monkeypatch.setattr("app.engine.search_stock_clip", stock_search)
    monkeypatch.setattr("app.engine._synthesize", lambda _text, output, *_args: output.write_bytes(b"audio"))
    monkeypatch.setattr("app.engine._probe_duration", lambda *_args: 8.0)
    monkeypatch.setattr("app.engine.choose_crop_anchor", lambda *_args: (0.5, 0.5))
    monkeypatch.setattr("app.engine._create_artwork", lambda _scene, _index, _ratio, output, *_args: output.write_bytes(b"image"))
    monkeypatch.setattr(
        "app.engine._render_video",
        lambda _visuals, _scenes, _audio, _ratio, output, *_args: output.write_bytes(b"rendered"),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={
                "scenes": [
                    {"text": "Consistency creates meaningful results over time."},
                    {"text": "A student studies with a laptop.", "visual_hint": "student using laptop"},
                ]
            },
            headers={"Idempotency-Key": "abstract-voiceover"},
        )
        completed = wait_for_status(client, response.json()["job_id"], "succeeded")
        timeline = client.get(completed["result"]["files"]["timeline.json"]).json()
        captions = client.get(completed["result"]["files"]["captions.srt"]).text

    assert stock_searches == ["scene-001"]
    assert not _is_visually_representable({"text": "Consistency creates meaningful results over time."})
    assert timeline["scenes"][0]["visual_status"] == "procedural_illustration"
    assert timeline["scenes"][0]["clip"]["source"] == "procedural-illustration"
    assert timeline["scenes"][0]["clip"]["fallback_used"] is True
    assert timeline["scenes"][0]["clip"]["provider_asset_id"] is None
    assert timeline["scenes"][0]["clip"]["reused_clip"] is False
    assert timeline["scenes"][1]["visual_status"] == "stock_broll"
    assert timeline["scenes"][1]["clip"]["provider_asset_id"] == "clip-abstract-support"
    assert timeline["scenes"][1]["clip"]["reused_clip"] is False
    assert "Consistency creates meaningful results over time." in captions


def test_provider_timeout_returns_within_budget_and_records_timeout(tmp_path, monkeypatch):
    from app.providers import MAX_TOTAL_SEARCH_SECONDS_PER_SCENE

    settings = replace(
        make_settings(tmp_path),
        visual_provider="wikimedia",
        visual_provider_order=("wikimedia",),
        internet_archive_enabled=False,
    )
    monkeypatch.setattr("app.providers.PROVIDER_TIMEOUT_SECONDS", 0.02)

    def slow_provider(*_args):
        time.sleep(0.2)
        return None

    started = time.monotonic()
    selected, error, timed_out = _call_provider_with_timeout(
        slow_provider,
        {"text": "A person works at a desk.", "visual_hint": "office work"},
        set(),
        tmp_path / "slow.candidate.mp4",
        settings,
        0.02,
    )
    elapsed = time.monotonic() - started

    assert timed_out is True
    assert selected is None
    assert isinstance(error, TimeoutError)
    assert elapsed < MAX_TOTAL_SEARCH_SECONDS_PER_SCENE


def test_stock_search_limits_remain_bounded():
    from app.providers import (
        MAX_PROVIDERS_PER_SCENE,
        MAX_QUERIES_PER_SCENE,
        MAX_TOTAL_SEARCH_SECONDS_PER_SCENE,
        PROVIDER_TIMEOUT_SECONDS,
    )

    assert MAX_QUERIES_PER_SCENE == 2
    assert MAX_PROVIDERS_PER_SCENE == 2
    assert PROVIDER_TIMEOUT_SECONDS == 4.0
    assert MAX_TOTAL_SEARCH_SECONDS_PER_SCENE == 6.0


def test_eleven_scenes_keep_scene_local_visuals_and_never_backfill_one_clip(monkeypatch, tmp_path):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="pixabay",
        pixabay_api_key="test-key",
    )
    app = create_app(settings, JobDatabase(settings.database_path), LocalStorage(settings.data_dir / "objects"))
    searched_scenes = []

    def fake_stock_search(scene, _used_ids, destination, _settings):
        searched_scenes.append(scene["_scene_id"])
        if scene["_scene_id"] != "scene-010":
            raise StockProvidersExhausted(
                [{"provider": "pixabay", "status": "no_relevant_clip", "reason": "no usable candidates"}]
            )
        subprocess.run(
            [
                settings.ffmpeg_path, "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=30",
                "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(destination),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return {
            "path": destination,
            "source": "Pixabay",
            "source_url": "https://pixabay.com/videos/example",
            "license": "Pixabay Content License",
            "attribution": "Pixabay contributor",
            "video_id": "clip-1",
            "semantic_similarity": 0.29,
            "relevance_score": 0.29,
            "fallback": False,
            "provider_attempts": [
                {"provider": "pixabay", "status": "selected", "reason": "verified"}
            ],
            "search_duration_ms": 10,
        }

    monkeypatch.setattr("app.engine.search_stock_clip", fake_stock_search)
    def make_audio(_text, output, *_args):
        subprocess.run(
            [
                settings.ffmpeg_path, "-y",
                "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
                "-t", "27.5", "-c:a", "pcm_s16le", str(output),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )

    monkeypatch.setattr("app.engine._synthesize", make_audio)
    monkeypatch.setattr("app.engine._probe_duration", lambda *_args: 27.5)
    monkeypatch.setattr("app.engine.choose_crop_anchor", lambda *_args: (0.5, 0.5))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/broll/jobs",
            json={
                "scenes": [
                    {
                        "text": f"A student uses a laptop for learning and research in scene {index}.",
                        "visual_hint": f"student learning on laptop scene {index}",
                    }
                    for index in range(11)
                ]
            },
            headers={"Idempotency-Key": "reuse-previous-visual"},
        )
        completed = wait_for_status(client, response.json()["job_id"], "succeeded", timeout=90)
        timeline = client.get(completed["result"]["files"]["timeline.json"]).json()
        captions = client.get(completed["result"]["files"]["captions.srt"]).text
        video_url = completed["result"]["files"]["video_16_9.mp4"]

    assert searched_scenes == [f"scene-{index:03}" for index in range(11)]
    assert len(timeline["scenes"]) == 11
    assert all(scene["narration"] == scene["text"] for scene in timeline["scenes"])
    assert all(scene["clip"]["reused_clip"] is False for scene in timeline["scenes"])
    assert all(scene["clip"]["fallback_used"] is True for scene in timeline["scenes"][:10])
    assert len({scene["visual_assets"]["16:9"]["source_path"] for scene in timeline["scenes"][:10]}) == 10
    assert timeline["scenes"][10]["visual_assets"]["16:9"]["source_path"] == "stock_010.mp4"
    assert all(
        scene["visual_assets"]["16:9"]["path"] == f"clip_16_9_{index:03}.mp4"
        for index, scene in enumerate(timeline["scenes"])
    )
    job_work_dir = settings.data_dir / "work" / response.json()["job_id"]
    for scene in timeline["scenes"]:
        visual_asset = scene["visual_assets"]["16:9"]
        segment_path = job_work_dir / visual_asset["path"]
        assert segment_path.is_file() and segment_path.stat().st_size > 0
        probe = subprocess.run(
            [
                settings.ffprobe_path,
                "-v", "error",
                "-show_entries", "stream=codec_type,codec_name",
                "-show_entries", "format=duration",
                "-of", "json",
                str(segment_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        media = json.loads(probe.stdout)
        assert any(
            stream.get("codec_type") == "video" and stream.get("codec_name") == "h264"
            for stream in media["streams"]
        )
        expected_scene_duration = scene["end"] - scene["start"]
        assert float(media["format"]["duration"]) == pytest.approx(expected_scene_duration, abs=0.15)
        assert visual_asset["duration"] == pytest.approx(expected_scene_duration, abs=0.001)
    final_probe = subprocess.run(
        [
            settings.ffprobe_path,
            "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(job_work_dir / "video_16_9.mp4"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    final_stream_types = {
        stream["codec_type"] for stream in json.loads(final_probe.stdout)["streams"]
    }
    assert {"audio", "video"} <= final_stream_types
    assert video_url.endswith("/video_16_9.mp4")
    assert all(scene["visual_assets"]["16:9"]["fallback_used"] for scene in timeline["scenes"][:10])
    assert timeline["scenes"][10]["visual_assets"]["16:9"]["fallback_used"] is False
    assert timeline["scenes"][10]["visual_status"] == "stock_broll"
    assert timeline["scenes"][10]["clip"]["provider"] == "Pixabay"
    assert timeline["scenes"][10]["clip"]["provider_asset_id"] == "clip-1"
    assert timeline["scenes"][10]["clip"]["fallback_used"] is False
    assert [scene["clip"]["provider_asset_id"] for scene in timeline["scenes"]] == [None] * 10 + ["clip-1"]
    assert "scene 10" in captions


def test_stock_verification_retains_verified_candidates_below_contextual_floor(tmp_path, monkeypatch):
    from app.providers import _verify_stock_file
    from app.relevance import MIN_CONTEXTUAL_FALLBACK_SIMILARITY, MIN_VISUAL_SIMILARITY

    settings = make_settings(tmp_path)
    score = MIN_VISUAL_SIMILARITY - 0.01
    monkeypatch.setattr(
        "app.providers._download_video",
        lambda _url, path: path.write_bytes(b"video"),
    )
    monkeypatch.setattr("app.providers._validate_video", lambda *_args: None)
    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: score)

    result = _verify_stock_file(
        "https://cdn.example.com/unrelated.mp4",
        {"text": "A gardener tends flowers in a garden.", "visual_hint": "gardening"},
        tmp_path / "result.mp4",
        settings,
        1,
    )

    assert result is not None
    assert result[0] == score
    assert result[1].is_file()
    result[1].unlink()

    score = MIN_CONTEXTUAL_FALLBACK_SIMILARITY - 0.01
    retained = _verify_stock_file(
        "https://cdn.example.com/unrelated.mp4",
        {"text": "A gardener tends flowers in a garden.", "visual_hint": "gardening"},
        tmp_path / "rejected.mp4",
        settings,
        2,
    )
    assert retained is not None
    assert retained[0] == score
    assert retained[1].is_file()
    retained[1].unlink()


def test_low_relevance_uses_highest_contextual_candidate_with_disclosure(tmp_path, monkeypatch):
    from app.relevance import MIN_VISUAL_SIMILARITY

    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        internet_archive_enabled=False,
        visual_provider_order=("wikimedia",),
    )

    def contextual_clip(_scene, _used_ids, destination, _settings):
        destination.write_bytes(b"contextual video")
        return {
            "path": destination,
            "source": "Wikimedia Commons",
            "source_url": "https://commons.wikimedia.org/wiki/File:Planning.mp4",
            "video_id": "contextual-clip",
            "semantic_similarity": MIN_VISUAL_SIMILARITY - 0.01,
            "title_match_score": 3,
            "selection_reason": "Matched task planning footage",
        }

    monkeypatch.setattr("app.providers.search_wikimedia_clip", contextual_clip)
    destination = tmp_path / "fallback.mp4"
    selected = search_stock_clip(
        {
            "text": "Organizing tasks at the beginning of the day can reduce distractions.",
            "visual_hint": "person planning daily tasks",
            "_scene_id": "scene-003",
        },
        set(),
        destination,
        settings,
    )

    assert selected["fallback"] is True
    assert selected["fallback_reason"] == "no_high_confidence_match"
    assert selected["relevance_score"] == MIN_VISUAL_SIMILARITY - 0.01
    assert destination.read_bytes() == b"contextual video"
    assert "fallback floor" in selected["selection_reason"]
    assert any(
        attempt["status"] == "selected"
        for attempt in selected["provider_attempts"]
    )


def test_last_resort_uses_best_verified_candidate_below_contextual_floor(tmp_path, monkeypatch):
    from app.relevance import MIN_CONTEXTUAL_FALLBACK_SIMILARITY

    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        internet_archive_enabled=False,
        visual_provider_order=("wikimedia",),
    )
    def verified_clip(_scene, _used_ids, destination, _settings):
        score = MIN_CONTEXTUAL_FALLBACK_SIMILARITY - 0.02
        destination.write_bytes(f"verified {score}".encode())
        return {
            "path": destination,
            "source": "Wikimedia Commons",
            "source_url": "https://commons.wikimedia.org/wiki/File:Planning.mp4",
            "video_id": f"clip-{score}",
            "semantic_similarity": score,
            "title_match_score": 3,
            "selection_reason": "Matched task planning footage",
        }

    monkeypatch.setattr("app.providers.search_wikimedia_clip", verified_clip)
    destination = tmp_path / "last-resort.mp4"
    selected = search_stock_clip(
        {
            "text": "Organizing tasks at the beginning of the day can reduce distractions.",
            "visual_hint": "person planning daily tasks",
            "_scene_id": "scene-003",
        },
        set(),
        destination,
        settings,
    )

    assert selected["fallback"] is True
    assert selected["fallback_reason"] == "low_confidence_contextual_match"
    assert selected["relevance_score"] == round(MIN_CONTEXTUAL_FALLBACK_SIMILARITY - 0.02, 4)
    assert destination.read_bytes() == f"verified {MIN_CONTEXTUAL_FALLBACK_SIMILARITY - 0.02}".encode()
    assert "last-resort contextual fallback" in selected["selection_reason"]
    assert "human relevance review" in selected["fallback_decision"]
    assert not list(tmp_path.glob("*.candidate.mp4"))


def test_stock_search_fails_when_no_provider_returns_a_usable_candidate(tmp_path, monkeypatch):
    settings = replace(
        make_settings(tmp_path),
        visual_provider="auto",
        visual_provider_order=("wikimedia",),
        internet_archive_enabled=False,
    )
    monkeypatch.setattr("app.providers.search_wikimedia_clip", lambda *_args: None)

    try:
        search_stock_clip(
            {"text": "A gardener tends flowers.", "visual_hint": "gardening flowers"},
            set(),
            tmp_path / "missing.mp4",
            settings,
        )
    except StockProvidersExhausted as exc:
        assert exc.attempts[0]["status"] == "no_relevant_clip"
    else:
        raise AssertionError("Expected stock search to fail when no usable candidates exist")


def test_groq_visual_review_returns_explicit_uncalibrated_boolean_verdict(tmp_path, monkeypatch):
    from PIL import Image

    settings = replace(
        make_settings(tmp_path),
        groq_api_key="test-key",
        groq_vision_model="test-vision-model",
    )
    captured = {}

    def fake_run(command, **_kwargs):
        Image.new("RGB", (16, 16), "blue").save(command[-1])
        return subprocess.CompletedProcess(command, 0)

    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {"relevant": True, "brand_logo_visible": False, "safety_concern": False}
                    )
                }
            }
        ]
    }

    def fake_urlopen(request, timeout):
        captured["request"] = json.loads(request.data)
        captured["authorization"] = request.get_header("Authorization")
        captured["timeout"] = timeout
        return BytesIO(json.dumps(response).encode())

    monkeypatch.setattr("app.visual_review.subprocess.run", fake_run)
    monkeypatch.setattr("app.visual_review.urllib.request.urlopen", fake_urlopen)
    from app.visual_review import review_candidate_frame

    verdict = review_candidate_frame(
        tmp_path / "candidate.mp4",
        {"text": "A nurse cares for a patient.", "visual_hint": "nurse in hospital"},
        settings,
    )

    assert verdict == {
        "status": "reviewed",
        "model": "test-vision-model",
        "relevant": True,
        "brand_logo_visible": False,
        "safety_concern": False,
        "calibrated": False,
    }
    assert captured["authorization"]
    assert captured["request"]["model"] == "test-vision-model"
    assert captured["request"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert captured["timeout"] == 30


def test_close_candidate_with_logo_flag_is_rejected(tmp_path, monkeypatch):
    from app.providers import _verify_stock_file

    settings = replace(
        make_settings(tmp_path),
        groq_api_key="test-key",
        groq_vision_model="test-vision-model",
    )
    monkeypatch.setattr("app.providers._download_video", lambda _url, path: path.write_bytes(b"video"))
    monkeypatch.setattr("app.providers._validate_video", lambda *_args: None)
    monkeypatch.setattr("app.providers.score_video_file", lambda *_args: 0.23)
    monkeypatch.setattr(
        "app.providers.review_candidate_frame",
        lambda *_args: {
            "status": "reviewed",
            "model": "test-vision-model",
            "relevant": True,
            "brand_logo_visible": True,
            "safety_concern": False,
            "calibrated": False,
        },
    )

    result = _verify_stock_file(
        "https://cdn.example.com/clip.mp4",
        {"text": "A gardener works in a garden.", "visual_hint": "gardening"},
        tmp_path / "result.mp4",
        settings,
        1,
    )

    assert result is None
    assert not (tmp_path / "result.candidate-1.mp4").exists()
