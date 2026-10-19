# IncuBrix B-roll Video Engine

An asynchronous FastAPI service that accepts a script, an audio URL, or an uploaded MP3/WAV/M4A file and returns a narrated video, SRT captions, a scene timeline, and a provenance manifest. Jobs use SQLite for status/idempotency and MinIO for artifacts in Compose.

## Current scope and limitations

Pexels stock search is the default visual source when `PEXELS_API_KEY` is configured. Results are filtered by clip-page title overlap, then the local OpenCLIP ViT-B/32 model scores three sampled frames against the narration and visual hint. Clips below cosine similarity `0.20` are rejected, and clip IDs are not reused within a job. This automated gate is a useful relevance check, not a replacement for the human-rated 85% evaluation.

If stock search is unavailable, returns no sufficiently matching title, or no clip passes the frame score, the scene falls back to a procedural illustration. The source, licence, attribution, clip link, AI-generation disclosure status, selection reason, and automated relevance estimate are included in `timeline.json`. The UI shows Pexels and photographer credits for stock results. OpenCLIP does not verify watermarks, safety, or possible AI-generated source footage; review those before publication. The model weights are downloaded on the first stock-scoring run and cached under the data directory.

Scripts use local eSpeak narration in the container. Audio-only jobs use local faster-whisper transcription; its model is downloaded on first use. Tests use the mock TTS provider and need no API credentials. The Pexels API is optional for local tests but required to exercise stock retrieval.

## Start

### Run locally on Windows

Python 3.12 and FFmpeg/FFprobe must be installed and available on `PATH`. The local vision extra installs PyTorch/OpenCLIP; its model weights are downloaded the first time a stock clip is scored.

```powershell
python -m pip install -e ".[test,asr,vision]"
$env:DATA_DIR = "$PWD\runtime-data"
$env:DATABASE_PATH = "$PWD\runtime-data\jobs.sqlite3"
$env:STORAGE_BACKEND = "local"
$env:TTS_PROVIDER = "pyttsx3"
$env:ASR_PROVIDER = "faster-whisper"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the video creator or `http://127.0.0.1:8000/docs` for the API. The first audio-only transcription downloads the free local Whisper model if it is not already cached. It can take a while and needs internet access once. Script-only jobs do not need ASR.

### Run with Docker Compose

1. Install Docker Compose.
2. Copy `.env.example` to `.env`; set `PEXELS_API_KEY` to enable stock search, and replace the development MinIO credentials.
3. Run `docker compose up --build`.
4. Open `http://localhost:8000`.

The API and MinIO data are stored in Compose volumes. The sample MinIO credentials are for local development only; replace them before exposing the service. Docker Compose has not yet been verified in this development environment.

## Hosting and GitHub

GitHub stores the source code and runs automated tests; it does not run this video API for reviewers. FFmpeg rendering, local narration, transcription, and persistent result files need a host with a writable disk and enough CPU/RAM. Free app hosts may sleep, limit processing time or storage, and remove local files. No free host can promise unlimited long-video processing while maintaining predictable quality.

This repository includes a GitHub Actions test workflow. Before publishing, create a private GitHub repository and push this project without `.env`, `runtime-data`, or credentials. Choose the review host after that; a small always-on server with persistent storage is the more reliable option for a time-limited evaluation. Never expose an unauthenticated demo to the public internet; this prototype still needs authentication and rate limiting before public deployment.

After creating an empty private repository on GitHub, connect and publish this local repository:

```powershell
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git add .
git commit -m "Build the IncuBrix B-roll demo"
git push -u origin main
```

Then add the reviewer as a collaborator in the GitHub repository settings. The host decision and public demo deployment remain separate steps; do not put API keys in GitHub source, commits, or Actions logs.

## Submit a job

```powershell
curl.exe -X POST http://localhost:8000/api/v1/broll/jobs `
  -H "Content-Type: application/json" `
  -H "Idempotency-Key: demo-001" `
  -d '{"script":"Saving starts with a plan. Track your spending.","aspect_ratios":["16:9","9:16"],"captions":{"burn_in":true,"srt":true}}'
```

For uploaded audio, send a `multipart/form-data` request with `audio_file`; JSON requests may instead set `audio_url`. Poll `GET /api/v1/jobs/{job_id}` until `succeeded` or `failed`; artifact links are returned in `result.files`.

For audio without a supplied script or scene list, faster-whisper transcribes the narration. To supply exact scene wording without ASR, send `scenes` alongside the audio; timing is then allocated proportionally to scene word counts.

## Tests and batch runner

```powershell
docker compose --profile test run --rm tests
python -m pip install -e ".[test]"
python -m pytest
python scripts/batch.py .\sample-inputs --base-url http://localhost:8000 --output-dir .\test-results
```

The batch runner accepts `.txt`, `.json`, `.mp3`, `.wav`, and `.m4a` files, polls each job, and saves its output artifacts and status under the result folder.

The ten-case human relevance pack is in [evaluation/](./evaluation/). Run it against the configured Pexels path, watch and rate every scene, then score the completed ratings CSV with `python scripts/evaluate_relevance.py evaluation/ratings-template.csv`. The template is intentionally blank; no relevance threshold is claimed until reviewers enter their ratings.

## API and settings

The checked-in `openapi.yaml` documents the API contract. FastAPI also serves a generated contract at `/openapi.json`.

All runtime settings are in `.env.example`. Important settings include:

| Setting | Purpose |
|---|---|
| `TTS_PROVIDER` | `espeak`, `pyttsx3`, or test-only `mock` |
| `VISUAL_PROVIDER` | `pexels` (default) or `procedural` |
| `PEXELS_API_KEY` | Pexels API key; required for stock results |
| `ASR_PROVIDER`, `ASR_MODEL`, `ASR_DEVICE` | Local faster-whisper transcription |
| `STORAGE_BACKEND` | `minio` or `local` |
| `MAX_AUDIO_SECONDS`, `MAX_UPLOAD_BYTES` | Input limits |
| `WORKER_COUNT` | Concurrent in-process render jobs |
| `MIN_SCENE_SECONDS` | Groups adjacent sentences when necessary to avoid very short shots |

API error codes include `IDEMPOTENCY_KEY_REQUIRED`, `IDEMPOTENCY_KEY_REUSED`, `INVALID_REQUEST`, `INVALID_JSON`, `UNSUPPORTED_CONTENT_TYPE`, `UNSUPPORTED_AUDIO_TYPE`, `INVALID_AUDIO_URL`, `INVALID_AUDIO`, `AUDIO_SOURCE_UNAVAILABLE`, `AUDIO_TOO_LARGE`, `AUDIO_TOO_LONG`, `MEDIA_TOOL_UNAVAILABLE`, `MEDIA_TOOL_TIMEOUT`, `MEDIA_PROCESSING_FAILED`, `TTS_UNAVAILABLE`, `TTS_FAILED`, `TTS_PROVIDER_UNSUPPORTED`, `ASR_UNAVAILABLE`, `ASR_PROVIDER_UNSUPPORTED`, `TRANSCRIPTION_FAILED`, `TRANSCRIPTION_EMPTY`, `EMPTY_SCENES`, `VISUAL_PROVIDER_UNSUPPORTED`, `JOB_PROCESSING_FAILED`, `JOB_NOT_FOUND`, `ARTIFACT_NOT_FOUND`, `ARTIFACT_STORAGE_UNAVAILABLE`, `NOT_READY`, and `INTERNAL_ERROR`.

## Artifacts and rights

Each result contains `video_16_9.mp4`, `video_9_16.mp4`, and/or `video_1_1.mp4` for the requested ratios, `captions.srt` when enabled, `timeline.json`, and `manifest.json`. The timeline records scene boundaries, provider asset ID, source, licence, commercial-use terms, attribution, clip link, AI-generation disclosure status, watermark and safety-review status, selection reason, OpenCLIP cosine similarity, and title-match score. These are automated signals, not the human 1–5 ratings required by the evaluation pack. The manifest records configured providers, settings, timings, a local cost estimate, warnings, and rights notes.

Procedural visuals are original graphics, not AI-generated imagery; their `ai_generated` flag is `false` and this service does not draw realistic faces or brand logos. For Pexels footage, the provider does not disclose AI-generation status in the response used here, so the timeline records it as unknown rather than asserting that the clip is not AI-generated. Pexels results and procedural fallbacks require human relevance and safety review before publication.
