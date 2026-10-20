# AI and model-use disclosure

- GitHub Copilot (Copilot SDK in VS Code) was used to inspect requirements, develop code, and validate the implementation.
- Kokoro-82M generates narration locally when `TTS_PROVIDER=kokoro`; the model weights and English phonemizer are downloaded on first use.
- faster-whisper is an optional local speech-recognition dependency. Audio files are not sent to a speech API by the default ASR path.
- OpenCLIP ViT-B/32 runs locally on candidate video frames and narration text to estimate scene relevance. It also chooses a semantic crop anchor from five frame crops. This is a heuristic; it does not prove human-perceived relevance or detect all unsafe content.
- When `GROQ_API_KEY` is configured, missing scene visual hints are sent to the configured Groq text model. When both `GROQ_API_KEY` and `GROQ_VISION_MODEL` are configured, a close-scoring stock candidate may send one extracted JPEG frame and the corresponding scene text to Groq for a relevance/logo/safety verdict. This visual verdict is not calibrated and must not be represented as a calibrated logo detector.
- In `auto` stock mode, narration-derived search terms are sent to whichever configured stock provider is queried (Pixabay, Openverse, Wikimedia Commons, or Pexels). Candidate video files are downloaded for local frame scoring and rendering.
- The visual path does not generate replacement footage with a generative video model. If real stock footage is unavailable, the narration continues over previously verified footage or a scene-specific procedural illustration; the output timeline identifies those fallbacks, which are not stock footage.

Do not submit confidential scripts or voice recordings to a public demo. Groq processing is optional, remote, and should be disabled for sensitive material.
