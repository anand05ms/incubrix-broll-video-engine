# Sources and third-party components

The default visual source is Pexels when a `PEXELS_API_KEY` is configured; otherwise, or when no sufficiently matching clip is found, the service falls back to original procedural illustrations. The stock path sends narration-derived search terms to Pexels.

| Component | Use | Licence / terms |
|---|---|---|
| Python | Runtime | PSF License |
| FastAPI, Starlette, Pydantic | HTTP API and request validation | MIT |
| Uvicorn | ASGI server | BSD-3-Clause |
| Pillow | Original procedural illustrations | HPND |
| FFmpeg | Audio/video processing | FFmpeg project licence; this image uses the distribution's LGPL/GPL build configuration |
| pyttsx3 | Optional local speech synthesis on supported platforms | MPL-2.0 |
| eSpeak NG | Local script narration | GPL-3.0; optional voice dependency, not linked into the Python service |
| faster-whisper | Optional local audio transcription in the container | MIT |
| CTranslate2 and Whisper model weights | faster-whisper runtime/model | Review each installed package and model card/licence before commercial deployment |
| MinIO Python SDK | S3-compatible result storage | Apache-2.0 |
| MinIO server image | Local S3-compatible object storage | GNU AGPLv3; development Compose dependency only, replace with a commercially approved S3-compatible service before product deployment |
| Pexels API and video library | Default stock-footage provider when configured | Pexels License; commercial use is subject to its terms. The UI links to Pexels and displays photographer/clip links. AI-generation and watermark status are not disclosed/verified by this implementation and require review. |
| OpenCLIP ViT-B/32 | Local frame-to-narration similarity ranking | OpenCLIP project license and pretrained-weight license; verify the exact selected model terms before redistribution |
| PyTorch and timm | OpenCLIP inference | BSD-style / Apache-2.0; verify installed distribution metadata before shipping |
| HTTPX and pytest | API test client and test runner | BSD-3-Clause / MIT |

Review upstream licences and current model terms before shipping or redistributing the container. The current implementation uses generated graphics and reports no third-party media in video timelines.
