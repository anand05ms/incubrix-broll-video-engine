from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    database_path: Path
    ffmpeg_path: str
    ffprobe_path: str
    tts_provider: str
    tts_voice: str
    visual_provider: str
    visual_provider_order: tuple[str, ...]
    internet_archive_enabled: bool
    require_stock_footage: bool
    pexels_api_key: str
    pixabay_api_key: str
    openverse_client_id: str
    openverse_client_secret: str
    groq_api_key: str
    groq_model: str
    groq_vision_model: str
    asr_provider: str
    asr_model: str
    asr_device: str
    asr_compute_type: str
    asr_translate_for_search: bool
    storage_backend: str
    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str
    s3_secure: bool
    max_audio_seconds: int
    max_upload_bytes: int
    worker_count: int
    job_execution_mode: str
    job_poll_interval: float
    callback_hmac_secret: str
    min_scene_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)
        data_dir = Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()
        database_path = Path(os.getenv("DATABASE_PATH", str(data_dir / "jobs.sqlite3"))).expanduser().resolve()
        tts_provider = os.getenv("TTS_PROVIDER", "kokoro").strip().lower()
        return cls(
            data_dir=data_dir,
            database_path=database_path,
            ffmpeg_path=os.getenv("FFMPEG_PATH", "ffmpeg"),
            ffprobe_path=os.getenv("FFPROBE_PATH", "ffprobe"),
            tts_provider=tts_provider,
            tts_voice=os.getenv("TTS_VOICE", "af_heart" if tts_provider == "kokoro" else "en"),
            visual_provider=os.getenv("VISUAL_PROVIDER", "auto").strip().lower(),
            visual_provider_order=tuple(
                provider.strip().lower()
                for provider in os.getenv(
                    "VISUAL_PROVIDER_ORDER",
                    "pixabay,openverse,wikimedia,internet_archive,pexels",
                ).split(",")
                if provider.strip()
            ),
            internet_archive_enabled=os.getenv("INTERNET_ARCHIVE_ENABLED", "true").lower() == "true",
            require_stock_footage=os.getenv("REQUIRE_STOCK_FOOTAGE", "true").lower() == "true",
            pexels_api_key=os.getenv("PEXELS_API_KEY", ""),
            pixabay_api_key=os.getenv("PIXABAY_API_KEY", ""),
            openverse_client_id=os.getenv("OPENVERSE_CLIENT_ID", ""),
            openverse_client_secret=os.getenv("OPENVERSE_CLIENT_SECRET", ""),
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            groq_vision_model=os.getenv("GROQ_VISION_MODEL", ""),
            asr_provider=os.getenv("ASR_PROVIDER", "faster-whisper"),
            asr_model=os.getenv("ASR_MODEL", "small"),
            asr_device=os.getenv("ASR_DEVICE", "cpu"),
            asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
            asr_translate_for_search=os.getenv("ASR_TRANSLATE_FOR_SEARCH", "true").lower() == "true",
            storage_backend=os.getenv("STORAGE_BACKEND", "local"),
            s3_endpoint=os.getenv("S3_ENDPOINT", "localhost:9000"),
            s3_bucket=os.getenv("S3_BUCKET", "broll-results"),
            s3_access_key=os.getenv("S3_ACCESS_KEY", ""),
            s3_secret_key=os.getenv("S3_SECRET_KEY", ""),
            s3_secure=os.getenv("S3_SECURE", "false").lower() == "true",
            max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "600")),
            max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024))),
            worker_count=max(1, int(os.getenv("WORKER_COUNT", "2"))),
            job_execution_mode=os.getenv("JOB_EXECUTION_MODE", "inline").strip().lower(),
            job_poll_interval=max(0.1, float(os.getenv("JOB_POLL_INTERVAL", "1"))),
            callback_hmac_secret=os.getenv("CALLBACK_HMAC_SECRET", ""),
            min_scene_seconds=max(0.5, float(os.getenv("MIN_SCENE_SECONDS", "2.5"))),
        )
