# Engineering Assessment 3 — Implementation Report

## Approach

The service is a Python 3.12/FastAPI application. SQLite persists job state and idempotency keys; a bounded thread pool performs rendering in the background. Artifacts are written to local storage for tests and to MinIO in the Compose deployment. FFmpeg normalizes output to H.264/AAC, 30 fps, and the requested aspect ratio.

Pexels is now the default visual provider when configured. Candidate clips are filtered by narration-query overlap with the clip-page title, then three frames from each of up to three candidates are scored against the narration with local OpenCLIP ViT-B/32. A candidate must exceed cosine similarity 0.20; selected clip IDs cannot be reused within a job. If no clip passes, the service falls back to an original procedural illustration and records a warning. Pexels AI-generation disclosure, watermark status, and content safety are not independently verified; the result timeline marks those limitations.

Scripts are narrated locally with eSpeak in the container. Audio-only inputs are transcribed with local faster-whisper; model weights download on first use. Explicit scene lists can be submitted with audio to avoid ASR.

## Verified in this workspace

- Fourteen pytest cases pass on Python 3.12.4 and FFmpeg 7.1.1 (Windows); the latest full run took 11.63 seconds.
- Tests exercise health/readiness, asynchronous script jobs, idempotent replay and conflict, audio-only upload validation, structured errors, artifact retrieval, portrait dimensions/frame rate, burned-in captions, stock-provider fallback, Pexels candidate ranking and uniqueness, relevance-score thresholding, and recovery of an interrupted job. The provider test mocks Pexels and frame scores.
- A real local OpenCLIP inference run on a synthetic bicycle video scored bicycle narration 0.2915 and unrelated pasta narration 0.1829 (cosine similarity; configured cutoff 0.20). This verifies the scorer runs and distinguishes this simple example, not its accuracy on real stock scenes.
- A ten-input script evaluation pack and a CSV-based scorer are included. The scorer enforces 85% of scenes rated 4 or 5, with all ten cases represented. No human ratings have been entered, so this is tooling only and does not establish that target.
- A browser UI is included for choosing script/audio, aspect ratio, style, avoid terms, captions, render progress, and output downloads.
- On 2026-10-19, a fresh Windows browser smoke test submitted the three-sentence balcony-garden story through the UI with local `pyttsx3` narration. The job succeeded in about 11 seconds and returned an MP4, SRT, timeline, and manifest. No Pexels key was configured, so both scenes used procedural illustrations; this verifies the local render flow only, not stock-footage relevance.
- The test TTS adapter produces a short silent audio fixture; it validates encoding and timing paths, not speech quality.
- A separate local smoke run verified actual `pyttsx3` narration and audio-only `faster-whisper` transcription. A 5.94-second WAV input completed transcription plus 1080p render in 38.48 seconds with the locally cached `base` model; the MP4 was verified as H.264/AAC at 30 fps. This is one cold local sample, not a representative benchmark.
- No Pexels key or Docker CLI was available, so neither live Pexels retrieval nor `docker compose up` was run here.

## Metrics not yet established

No representative one-minute narration corpus or human ratings were available. Turnaround per finished minute, production TTS/ASR cost, peak memory, concurrency capacity, coverage, and relevance agreement are **not measured**. Test costs are zero because tests use local code and a mock TTS fixture; that is not a production cost estimate.

## Known gaps

- OpenCLIP frame scoring is now wired into candidate selection, but its 0.20 threshold is only smoke-tested and the ten-case 85% human relevance target remains unverified.
- Pexels search/download and real OpenCLIP inference need a credentialed integration run, licence review, and human review for relevance, watermarks, safety, and AI-generated footage status before production use.
- There is no independent vision-language scoring, completed human relevance study, or verified 85% scene rating.
- There is no scene-level clip replacement endpoint, callback signing, authentication/rate limiting, or distributed job broker. The worker pool is in-process; SQLite job recovery is for a single service instance.
- ASR segment timestamps are retained with silence gaps assigned to adjacent scenes. Other scene timing is proportional to word count. This is not forced word-level alignment.
- Hindi/Tamil voice and transcription quality, sensitive-topic review, visual-content moderation, and commercial production metrics are unverified.
- Compose includes MinIO for local development; its AGPLv3 server image and the GPL eSpeak package require an explicit distribution/licensing decision before shipping.

## Recommended next steps

Configure Pexels and run the ten-case evaluation pack; enter human ratings and review every clip's source, licence, attribution, watermark, AI-generation disclosure, and safety. Add independent frame-level relevance/safety validation, then rerun the pack and verify the 85% threshold. Run Compose in a Docker-enabled environment and exercise actual eSpeak and audio-only transcription. Authentication, request rate limits, and deployment to a secured host are still required before exposing a public demo.
