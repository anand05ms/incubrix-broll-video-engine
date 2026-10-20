# IncuBrix B-roll Video Engine

An asynchronous FastAPI service that accepts a script, an audio URL, or an uploaded MP3/WAV/M4A file and returns a narrated video, SRT captions, a scene timeline, and a provenance manifest. SQLAlchemy persists jobs and idempotency state in SQLite; MinIO stores artifacts in Compose.

## Current scope and limitations

The video workflow is script/audio -> narration or transcription -> scene planning -> stock search -> OpenCLIP frame relevance scoring -> semantic crop framing -> FFmpeg render -> captions, timeline, and provenance manifest. Each scene keeps its own verified stock selection and FFmpeg visual input; selected stock clips are tracked across the job to prevent accidental duplicate use. Missing or abstract scenes receive their own text-free, scene-specific procedural illustration rather than inheriting a clip from another scene. These illustrations use an opaque, full-frame environment and recognizable scene props (for example, a workstation, laptop interface, book, or task board); they are original illustrations, not stock footage. Narration and captions continue when stock footage is unavailable. The timeline records each aspect ratio's actual rendered segment and input visual path. Before a scene is concatenated, its segment is checked for H.264 video, 30 fps, expected duration, and decodable visible frames; a failed source/render is replaced with a generated fallback segment. `VISUAL_PROVIDER=procedural` remains available when an explicitly generated-art preview is desired. Search is bounded to two generated queries, five results per query, at most two configured credential-ready providers, four seconds per provider, and six seconds total per scene. Pixabay is first in the default provider order; a verified result stops additional provider searches for that scene. Provider adapters retain their media, license, download, and relevance checks. Stock search can try Pixabay, Openverse, Wikimedia Commons, Internet Archive, and Pexels in configurable order. A named provider is preferred, not a lock. Keyed providers without credentials are skipped; set their environment variables (such as `PIXABAY_API_KEY`) to enable them. Wikimedia Commons and Internet Archive do not require API keys. Internet Archive is an additional no-cost catalog, filtered to verified CC BY, CC BY-SA, CC0, or Public Domain Mark declarations and media under the download limit. Openverse and Pixabay are no-cost options with provider registration/credentials; Pexels is optional and not required. Pixabay results are cached for 24 hours as required by its API terms. Provider/query attempts, search duration, failures, visual status, and fallback choices are recorded in result warnings, `manifest.json`, and `timeline.json`. All adapters depend on the providers' live catalogs and API availability; bounded search cannot guarantee a stock match for every possible story.

When Groq is unavailable, deterministic scene rules produce short, concrete visual-search hints rather than submitting narration fragments as queries. Burned-in captions use white text, a dark outline, and a semi-transparent dark backing for contrast over both stock footage and light procedural artwork; downloadable SRT captions remain unchanged.

Normal stock selections must clear the local OpenCLIP `0.20` threshold. If none do, a candidate scoring at least `0.17` may be used as a contextual fallback, marked with `fallback: true`, `fallback_reason: "no_high_confidence_match"`, and its `relevance_score` in `timeline.json`. If all verified candidates score below `0.17`, the highest-scoring verified candidate is selected as a last-resort contextual fallback and marked with `fallback_reason: "low_confidence_contextual_match"`; `manifest.json` includes the `LOW_CONFIDENCE_BROLL_FALLBACK` warning. Both fallback tiers retain license/media checks and any configured close-call review, but are not calibrated evidence of human-rated relevance and must be inspected before use. A missing scene keeps its own scene-specific text-free illustration rather than inheriting another scene's stock selection; narration continues. Before a result is marked complete, the renderer verifies each illustration is decodable and nonblank and checks that the final MP4 contains audio/video streams, expected duration, and sampled nonblack frames. OpenCLIP similarity is not the human-rated 85% evaluation. Watermarks, AI-generation status, safety, and logo absence are not universally verified; review footage before publication. When Groq vision is configured, close-scoring frames may receive a separate relevance/logo/safety review. Those model verdicts are un-calibrated and are not a substitute for human review or a calibrated logo detector.

Scripts use Kokoro-82M by default, with `af_heart` English and `hf_alpha` Hindi voice identifiers; install the `kokoro` extra. eSpeak and pyttsx3 remain configurable alternatives. Audio-only jobs use faster-whisper (`small`, `int8` by default); model weights download on first use. Groq (`openai/gpt-oss-120b`) can generate missing visual hints; an optional configured Groq vision model judges close calls. These calls send the relevant scene text, and vision review also sends a sampled frame to Groq. Offline scene rules and OpenCLIP-only screening are used when Groq is not configured. Model use and unknown costs are disclosed in the manifest.

SQLAlchemy stores durable job records in SQLite. Compose runs a separate queue worker that atomically claims jobs from the shared SQLite database; this is a single-host/single-worker deployment pattern, not a distributed broker or horizontally scalable database. MinIO stores artifacts in Compose. Optional terminal callbacks use HMAC-SHA256 signatures and require `CALLBACK_HMAC_SECRET`. The Docker image includes FFmpeg, eSpeak NG, and Noto fonts for multilingual captions. The app still has no authentication or rate limiting and must not be exposed publicly without a protective access layer.

## Start

### Run locally on Windows

Python 3.12 and FFmpeg/FFprobe must be installed and available on `PATH`. The app and worker load a project-root `.env` file at startup; values already set in the process environment take precedence. The `vision` extra installs PyTorch/OpenCLIP; `kokoro` installs Kokoro TTS and SoundFile; the ASR extra installs faster-whisper. Model weights are downloaded on first use.

The default `auto` stock mode tries configured providers from `VISUAL_PROVIDER_ORDER`; Internet Archive is automatically appended as a no-key fallback unless `INTERNET_ARCHIVE_ENABLED=false`. Wikimedia Commons also works without a key. Optional API credentials are configured in the environment; never paste them into chat or commit them:

```powershell
$env:PIXABAY_API_KEY = "YOUR_PIXABAY_API_KEY"
$env:OPENVERSE_CLIENT_ID = "YOUR_OPENVERSE_CLIENT_ID"
$env:OPENVERSE_CLIENT_SECRET = "YOUR_OPENVERSE_CLIENT_SECRET"
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
$env:GROQ_VISION_MODEL = "qwen/qwen3.8-27b"
```

```powershell
python -m pip install -e ".[test,asr,vision,kokoro]"
$env:DATA_DIR = "$PWD\runtime-data"
$env:DATABASE_PATH = "$PWD\runtime-data\jobs.sqlite3"
$env:STORAGE_BACKEND = "local"
$env:TTS_PROVIDER = "kokoro"
$env:ASR_PROVIDER = "faster-whisper"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` for the video creator or `http://127.0.0.1:8000/docs` for the API. Kokoro's model and English phonemizer model download on first use and require internet access once. Script-only jobs do not need ASR.
When renderer code changes, stop and restart the local Uvicorn process before submitting a new job; completed MP4 files are not regenerated automatically.

### Run with Docker Compose

1. Install Docker Compose.
2. Copy `.env.example` to `.env`; replace the development MinIO credentials. Pixabay/Openverse/Groq credentials improve search and review but are optional because Wikimedia Commons remains a no-key fallback.
3. Run `docker compose up --build`.
4. Open `http://localhost:8000`.

The API and separate worker share a Compose volume containing the SQLite job queue, model cache, and work files; MinIO stores deliverable artifacts. Keep exactly one worker with SQLite. Rebuild and restart both services after changing renderer code (`docker compose up --build -d api worker`). The sample MinIO credentials are for local development only; replace them before exposing the service. Docker Compose could not be run in this development environment.

## Hosting and GitHub

GitHub stores the source code and runs automated tests; it does not run this video API for reviewers. FFmpeg rendering, Kokoro, transcription, OpenCLIP, and persistent result files need a host with a writable volume and enough CPU/RAM. This repository currently has a Docker Compose deployment, not a verified Modal/Render/Vercel production deployment. Vercel is not a fit for long-running FFmpeg workers. A free-tier host may sleep, limit processing/storage, or remove files; no free host can guarantee unlimited rendering.

This repository includes a GitHub Actions test workflow. Before publishing, create a private GitHub repository and push this project without `.env`, `runtime-data`, or credentials. A deployment requires a persistent volume and an access layer; no public deployment was made or verified. Never expose the unauthenticated demo directly to the public internet.

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

The ten-case human relevance pack is in [evaluation/](./evaluation/). Run it against the configured auto-provider path, watch and rate every scene, then score the completed ratings CSV with `python scripts/evaluate_relevance.py evaluation/ratings-template.csv`. The template is intentionally blank; no relevance threshold is claimed until reviewers enter their ratings.

## API and settings

The checked-in `openapi.yaml` documents the API contract. FastAPI also serves a generated contract at `/openapi.json`.

All runtime settings are in `.env.example`. Important settings include:

| Setting | Purpose |
|---|---|
| `TTS_PROVIDER`, `TTS_VOICE` | `kokoro` and `af_heart` by default; also `espeak`, `pyttsx3`, and test-only `mock` |
| `VISUAL_PROVIDER` | `auto` (default), `pixabay`, `openverse`, `wikimedia`, `internet_archive`, `pexels`, or `procedural`; a named provider is preferred, with configured providers tried afterward |
| `VISUAL_PROVIDER_ORDER` | Fallback order used by `auto` and after the preferred named provider; Internet Archive is appended if omitted |
| `INTERNET_ARCHIVE_ENABLED` | `true` by default; enables the no-key Internet Archive fallback |
| `REQUIRE_STOCK_FOOTAGE` | Retained in job settings for compatibility; scenes without verified stock use a procedural illustration so rendering can continue |
| `PIXABAY_API_KEY` | Optional; enables Pixabay video search; API search results are cached for 24 hours |
| `OPENVERSE_CLIENT_ID`, `OPENVERSE_CLIENT_SECRET` | Optional no-cost Openverse API credentials; enables licensed video search |
| `PEXELS_API_KEY` | Optional; enables Pexels search |
| `GROQ_API_KEY`, `GROQ_MODEL` | Optional scene-to-visual-hint generation; otherwise offline rules are used |
| `GROQ_VISION_MODEL` | Optional Groq vision judge for close-scoring candidate frames; sends one sampled frame and scene text; verdicts are not calibrated |
| `ASR_PROVIDER`, `ASR_MODEL`, `ASR_DEVICE` | Local faster-whisper transcription |
| `ASR_COMPUTE_TYPE` | faster-whisper compute type; defaults to `int8` |
| `STORAGE_BACKEND` | `minio` or `local` |
| `MAX_AUDIO_SECONDS`, `MAX_UPLOAD_BYTES` | Input limits |
| `JOB_EXECUTION_MODE` | `inline` for local development; `external` for API + separate worker |
| `CALLBACK_HMAC_SECRET` | Required to send signed callbacks when a request includes `callback_url` |
| `WORKER_COUNT` | Concurrent inline render jobs; Compose uses one queue worker |
| `MIN_SCENE_SECONDS` | Groups adjacent sentences when necessary to avoid very short shots |

API error codes include `IDEMPOTENCY_KEY_REQUIRED`, `IDEMPOTENCY_KEY_REUSED`, `INVALID_REQUEST`, `INVALID_JSON`, `UNSUPPORTED_CONTENT_TYPE`, `UNSUPPORTED_AUDIO_TYPE`, `INVALID_AUDIO_URL`, `INVALID_AUDIO`, `AUDIO_SOURCE_UNAVAILABLE`, `AUDIO_TOO_LARGE`, `AUDIO_TOO_LONG`, `MEDIA_TOOL_UNAVAILABLE`, `MEDIA_TOOL_TIMEOUT`, `MEDIA_PROCESSING_FAILED`, `TTS_UNAVAILABLE`, `TTS_FAILED`, `TTS_PROVIDER_UNSUPPORTED`, `ASR_UNAVAILABLE`, `ASR_PROVIDER_UNSUPPORTED`, `TRANSCRIPTION_FAILED`, `TRANSCRIPTION_EMPTY`, `EMPTY_SCENES`, `VISUAL_PROVIDER_UNSUPPORTED`, `JOB_PROCESSING_FAILED`, `JOB_NOT_FOUND`, `ARTIFACT_NOT_FOUND`, `ARTIFACT_STORAGE_UNAVAILABLE`, `NOT_READY`, and `INTERNAL_ERROR`.

## Artifacts and rights

Each result contains the requested `video_16_9.mp4`, `video_9_16.mp4`, and/or `video_1_1.mp4`, `captions.srt` when enabled, `timeline.json`, and `manifest.json`. The timeline records scene boundaries, source asset, license, attribution, source disclosure, human-review caveats, selection reason, OpenCLIP similarity, and title-match score. A Groq verdict, if made, is explicitly marked un-calibrated. The manifest records providers, settings, timings, warnings, rights notes, and unknown cost when Groq was called.

Procedural visuals are original graphics, not AI-generated imagery; their `ai_generated` flag is `false`. Third-party stock AI-generation status and watermarks may be unknown. Review all stock scenes and current license terms before publication.
