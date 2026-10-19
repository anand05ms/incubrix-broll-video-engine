# AI tool disclosure

- GitHub Copilot (Copilot SDK in VS Code): used to inspect the supplied assessment requirements and generate/validate this implementation.
- When Pexels is configured, OpenCLIP ViT-B/32 (pretrained OpenAI CLIP weights) runs locally on three sampled frames per candidate and the narration text to estimate visual relevance. The weights download from the Hugging Face Hub on first use; inference runs on the host CPU, and candidate frames are not sent to an AI inference API.
- The visual path does not generate footage with a generative AI model. If stock matching is unavailable or fails its score threshold, the service draws original procedural illustrations.
- faster-whisper is an optional local speech-recognition dependency; it is not used for script-only jobs.
- When Pexels is enabled, narration-derived search terms are sent to the Pexels API and stock clips are downloaded from Pexels.
