# Sources and third-party components

The application searches Pixabay, Openverse, Wikimedia Commons, Internet Archive, and Pexels in configurable order. Keyed providers are skipped if their credentials are absent; Wikimedia Commons and Internet Archive are no-key sources. A named `VISUAL_PROVIDER` is a preference, not a single-provider lock: the remaining configured providers are tried after it. Internet Archive assets are accepted only with CC BY, CC BY-SA, CC0, or Public Domain Mark metadata and supported-size original video files; their frames still must pass the normal OpenCLIP relevance threshold. Selected assets and each attempted provider's outcome/fallback decision are recorded in warnings, `manifest.json`, and `timeline.json`. Automated license filters are not legal advice; verify current provider terms and every clip before commercial publication.

| Component | Use | Licence / terms |
|---|---|---|
| Python | Runtime | PSF License |
| FastAPI, Starlette, Pydantic | HTTP API and request validation | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| SQLAlchemy | Persistent job queue and SQLite ORM | MIT |
| Pillow | Procedural graphics and image/font rendering | HPND |
| FFmpeg / FFprobe | Audio/video processing, frame extraction, rendering | FFmpeg project licence; this image uses the distribution's LGPL/GPL build configuration |
| Kokoro-82M and Kokoro inference library | Local script narration | Apache-2.0 library and model weights; voice/model assets are downloaded on first use |
| pyttsx3 | Optional local speech synthesis fallback | MPL-2.0 |
| eSpeak NG | Optional local script narration / phoneme support | GPL-3.0; review distribution obligations |
| faster-whisper | Optional local audio transcription | MIT; separately review CTranslate2, model card, and weight terms |
| Pixabay video API | Optional stock-video search | Pixabay Content License and API terms; results must be cached for at least 24 hours; API key required |
| Openverse video API | Optional Creative Commons/public-domain video discovery | Per-asset license and Openverse API terms; no-cost API client registration/credentials required |
| Wikimedia Commons | No-key stock-video discovery | Per-asset license; only recognized public-domain, CC0, CC BY, and CC BY-SA entries are accepted |
| Internet Archive Advanced Search and Metadata APIs | No-key licensed-video discovery fallback | Per-item metadata must declare CC BY, CC BY-SA, CC0, or Public Domain Mark; verify the item's metadata and terms before use |
| Pexels video API | Optional stock-video search | Pexels License and API terms; API key required |
| OpenCLIP ViT-B/32 and pretrained weights | Local scene relevance scoring and semantic crop anchoring | Review the OpenCLIP package and selected pretrained-weight terms before redistribution |
| Groq API / configured models | Optional scene hints and close-call visual review | Groq account/model terms and billing apply; narration-derived text and, for visual review, one sampled frame are sent remotely |
| PyTorch, torchvision, timm | OpenCLIP/Kokoro inference dependencies | Review installed distribution licenses and notices |
| MinIO Python SDK | S3-compatible result storage | Apache-2.0 |
| MinIO server image | Local S3-compatible storage for Compose | GNU AGPLv3; development Compose dependency only unless distribution obligations are addressed |
| Noto fonts | Docker captions and Tamil/Devanagari text rendering | SIL Open Font License |
| HTTPX and pytest | API test client and test runner | BSD-3-Clause / MIT |

This project does not include a trained or calibrated logo-probability model. Optional Groq vision verdicts are heuristic, un-calibrated signals; human review remains necessary. Review upstream licenses, model cards, and provider terms before distributing or monetizing outputs.
