from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    ffmpeg_path: str
    ffprobe_path: str
    tts_provider: str
    tts_voice: str
    visual_provider: str
    pexels_api_key: str
    asr_provider: str
    asr_model: str
    asr_device: str
    storage_backend: str
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_secure: bool
    max_audio_seconds: int
    max_upload_bytes: int
    worker_count: int
    min_scene_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
        database_path = Path(os.getenv("DATABASE_PATH", str(data_dir / "jobs.sqlite3"))).expanduser().resolve()
        return cls(
            data_dir=data_dir,
            database_path=database_path,
            ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
            ffprobe_path=os.getenv("FFPROBE_PATH", "ffprobe"),
            tts_provider=os.getenv("TTS_PROVIDER", "espeak"),
            tts_voice=os.getenv("TTS_VOICE", "en"),
            visual_provider=os.getenv("VISUAL_PROVIDER", "pexels"),
            pexels_api_key=os.getenv("PEXELS_API_KEY", ""),
            asr_provider=os.getenv("ASR_PROVIDER", "faster-whisper"),
            asr_model=os.getenv("ASR_MODEL", "base"),
            asr_device=os.getenv("ASR_DEVICE", "cpu"),
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            s3_endpoint=os.getenv("S3_ENDPOINT", "localhost:9000"),
            s3_bucket=os.getenv("S3_BUCKET", "broll-results"),
            s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
            s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
            s3_secure=os.getenv("S3_SECURE", "false").lower() == "true",
            max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "600")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            worker_count=max(1, int(os.getenv("WORKER_COUNT", "2"))),
            min_scene_seconds=max(0.5, float(os.getenv("MIN_SCENE_SECONDS", "2.5"))),
        )
