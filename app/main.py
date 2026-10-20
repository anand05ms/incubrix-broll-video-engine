import hashlib
import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, ValidationError, ValidationInfo, field_validator, model_validator
from starlette.datastructures import UploadFile
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError

from app.config import Settings
from app.callbacks import validate_callback_url
from app.db import JobDatabase
from app.engine import ALLOWED_AUDIO_SUFFIXES, ALLOWED_RATIOS, EngineError
from app.jobs import execute_job
from app.logging_config import configure_logging
from app.storage import ObjectStorage, create_storage


logger = configure_logging()


class SceneInput(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    visual_hint: str | None = Field(default=None, max_length=1000)


class CaptionOptions(BaseModel):
    burn_in: bool = False
    srt: bool = True


class JobInput(BaseModel):
    audio_url: str | None = None
    script: str | None = Field(default=None, max_length=30000)
    scenes: list[SceneInput] | None = Field(default=None, max_length=100)
    language: Literal["en", "hi", "ta"] | None = None
    voice: str | None = Field(default=None, max_length=100)
    aspect_ratios: list[str] = Field(default_factory=lambda: ["16:9"], min_length=1, max_length=3)
    style: str | None = Field(default=None, max_length=500)
    avoid: list[str] = Field(default_factory=list, max_length=50)
    captions: CaptionOptions = Field(default_factory=CaptionOptions)
    callback_url: str | None = Field(default=None, max_length=2048)

    @field_validator("callback_url")
    @classmethod
    def validate_callback(cls, value: str | None) -> str | None:
        if value:
            validate_callback_url(value)
        return value

    @field_validator("aspect_ratios")
    @classmethod
    def validate_ratios(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(ratio not in ALLOWED_RATIOS for ratio in value):
            raise ValueError("aspect_ratios must contain unique values from 16:9, 9:16, and 1:1")
        return value

    @model_validator(mode="after")
    def validate_content(self, info: ValidationInfo) -> "JobInput":
        if self.script is not None and not self.script.strip():
            raise ValueError("script must not be empty")
        has_uploaded_audio = bool((info.context or {}).get("uploaded_audio"))
        if self.script is None and not self.audio_url and not self.scenes and not has_uploaded_audio:
            raise ValueError("Provide audio_url, script, uploaded audio, or scenes")
        if self.scenes is not None and not self.scenes:
            raise ValueError("scenes must contain at least one scene")
        return self


class APIError(Exception):
    def __init__(self, status_code: int, code: str, message: str, retryable: bool = False, details: Any = None):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        self.details = details


def _error(code: str, message: str, retryable: bool = False, details: Any = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "retryable": retryable, "details": details}}


def _serializable_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(errors, default=str))


def _parse_json_request(body: Any) -> tuple[JobInput, bytes | None, str | None]:
    if not isinstance(body, dict):
        raise APIError(400, "INVALID_REQUEST", "Request body must be a JSON object")
    file_data = body.pop("_uploaded_audio", None)
    filename = body.pop("_uploaded_filename", None)
    try:
        return JobInput.model_validate(body), file_data, filename
    except ValidationError as exc:
        raise APIError(
            422,
            "INVALID_REQUEST",
            "Request validation failed",
            details=_serializable_errors(exc.errors(include_input=False)),
        ) from exc


async def _read_request(request: Request, max_upload_bytes: int) -> tuple[JobInput, bytes | None, str | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise APIError(400, "INVALID_JSON", "Request body is not valid JSON") from exc
        return _parse_json_request(body)
    if content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
            file_obj = form.get("audio_file")
            audio_data = None
            filename = None
            if isinstance(file_obj, UploadFile):
                audio_data = await file_obj.read(max_upload_bytes + 1)
                filename = file_obj.filename or "audio"
                if len(audio_data) > max_upload_bytes:
                    raise APIError(413, "AUDIO_TOO_LARGE", f"Audio exceeds the {max_upload_bytes}-byte upload limit")
                if Path(filename).suffix.lower() not in ALLOWED_AUDIO_SUFFIXES:
                    raise APIError(415, "UNSUPPORTED_AUDIO_TYPE", "Accepted audio types are mp3, wav, and m4a")

            ratios = form.get("aspect_ratios", '["16:9"]')
            scenes = form.get("scenes")
            captions = form.get("captions")
            body = {
                "audio_url": form.get("audio_url") or None,
                "script": form.get("script") or None,
                "scenes": json.loads(scenes) if scenes else None,
                "language": form.get("language") or None,
                "voice": form.get("voice") or None,
                "aspect_ratios": json.loads(ratios) if ratios.startswith("[") else [part.strip() for part in ratios.split(",")],
                "style": form.get("style") or None,
                "avoid": json.loads(form.get("avoid", "[]")),
                "captions": json.loads(captions) if captions else {"burn_in": False, "srt": True},
                "callback_url": form.get("callback_url") or None,
            }
        except APIError:
            raise
        except (ValueError, json.JSONDecodeError) as exc:
            raise APIError(400, "INVALID_REQUEST", "Multipart JSON fields are malformed") from exc
        try:
            return JobInput.model_validate(body, context={"uploaded_audio": audio_data is not None}), audio_data, filename
        except ValidationError as exc:
            raise APIError(
                422,
                "INVALID_REQUEST",
                "Request validation failed",
                details=_serializable_errors(exc.errors(include_input=False)),
            ) from exc
    raise APIError(415, "UNSUPPORTED_CONTENT_TYPE", "Use application/json or multipart/form-data")


def _public_job(row: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(row["result_json"]) if row["result_json"] else None
    error = json.loads(row["error_json"]) if row["error_json"] else None
    return {
        "job_id": row["id"],
        "status": row["status"],
        "progress": row["progress"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "result": result,
        "error": error,
    }


def create_app(
    settings: Settings | None = None,
    database: JobDatabase | None = None,
    storage: ObjectStorage | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()
    job_db = database or JobDatabase(configured.database_path)
    object_storage = storage or create_storage(configured)
    if configured.job_execution_mode not in {"inline", "external"}:
        raise ValueError("JOB_EXECUTION_MODE must be 'inline' or 'external'")
    executor = (
        ThreadPoolExecutor(max_workers=configured.worker_count, thread_name_prefix="broll-job")
        if configured.job_execution_mode == "inline"
        else None
    )
    configured.data_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        job_db.initialize()
        if executor:
            for row in job_db.requeue_interrupted():
                executor.submit(
                    execute_job,
                    row["id"],
                    json.loads(row["request_json"]),
                    row["upload_path"],
                    row["correlation_id"],
                    job_db,
                    configured,
                    object_storage,
                )
        try:
            yield
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=False)
            job_db.close()

    app = FastAPI(
        title="IncuBrix Script and Voiceover to B-roll Video Engine",
        version="0.1.0",
        description="Asynchronous API for generating captioned videos from narration or scripts.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

    @app.middleware("http")
    async def add_correlation_id(request: Request, call_next: Any) -> Response:
        correlation_id = request.headers.get("X-Correlation-Id") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-Id"] = correlation_id
        return response

    @app.exception_handler(APIError)
    async def handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_error(exc.code, exc.message, exc.retryable, exc.details))

    @app.exception_handler(EngineError)
    async def handle_engine_error(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_error(exc.code, exc.message, exc.retryable))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error("INVALID_REQUEST", "Request validation failed", details=_serializable_errors(exc.errors())),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled API error",
            extra={"job_id": None, "correlation_id": getattr(request.state, "correlation_id", None)},
        )
        return JSONResponse(status_code=500, content=_error("INTERNAL_ERROR", "An unexpected server error occurred"))

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    async def landing() -> FileResponse:
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    @app.get("/ready", tags=["system"])
    async def ready() -> JSONResponse:
        try:
            job_db.initialize()
            object_storage.ensure_ready()
        except Exception as exc:
            return JSONResponse(status_code=503, content=_error("NOT_READY", str(exc), retryable=True))
        return JSONResponse(content={"status": "ready"})

    @app.post("/api/v1/broll/jobs", status_code=202, tags=["jobs"])
    async def create_job(
        request: Request,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> JSONResponse:
        if not idempotency_key or len(idempotency_key) > 200 or not re.fullmatch(r"[A-Za-z0-9._:-]+", idempotency_key):
            raise APIError(400, "IDEMPOTENCY_KEY_REQUIRED", "Provide an Idempotency-Key header (1-200 safe characters)")
        payload, audio_bytes, filename = await _read_request(request, configured.max_upload_bytes)
        if payload.callback_url and not configured.callback_hmac_secret:
            raise APIError(
                503,
                "CALLBACK_NOT_CONFIGURED",
                "Signed callbacks require CALLBACK_HMAC_SECRET to be configured",
            )
        request_data = payload.model_dump(mode="json")
        fingerprint = {
            "request": request_data,
            "upload_sha256": hashlib.sha256(audio_bytes).hexdigest() if audio_bytes is not None else None,
        }
        request_hash = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        previous = job_db.by_idempotency_key(idempotency_key)
        if previous:
            if previous["request_hash"] != request_hash:
                raise APIError(409, "IDEMPOTENCY_KEY_REUSED", "This Idempotency-Key was already used for a different request")
            return JSONResponse(status_code=202, content=_public_job(previous))

        job_id = str(uuid.uuid4())
        upload_path = None
        if audio_bytes is not None:
            extension = Path(filename or "").suffix.lower()
            upload_path_obj = configured.data_dir / "work" / job_id / f"input{extension}"
            upload_path_obj.parent.mkdir(parents=True, exist_ok=True)
            upload_path_obj.write_bytes(audio_bytes)
            upload_path = str(upload_path_obj)
        try:
            job_db.create(
                job_id,
                idempotency_key,
                request_hash,
                request_data,
                upload_path,
                request.state.correlation_id,
            )
        except IntegrityError:
            duplicate = job_db.by_idempotency_key(idempotency_key)
            if upload_path:
                Path(upload_path).unlink(missing_ok=True)
            if duplicate:
                if duplicate["request_hash"] == request_hash:
                    return JSONResponse(status_code=202, content=_public_job(duplicate))
                raise APIError(409, "IDEMPOTENCY_KEY_REUSED", "This Idempotency-Key was already used for a different request")
            raise
        if executor:
            executor.submit(
                execute_job,
                job_id,
                request_data,
                upload_path,
                request.state.correlation_id,
                job_db,
                configured,
                object_storage,
            )
        row = job_db.get(job_id)
        return JSONResponse(status_code=202, content=_public_job(row))

    @app.get("/api/v1/jobs/{job_id}", tags=["jobs"])
    async def get_job(job_id: str) -> dict[str, Any]:
        row = job_db.get(job_id)
        if not row:
            raise APIError(404, "JOB_NOT_FOUND", "No job exists with this job_id")
        return _public_job(row)

    @app.get("/api/v1/jobs/{job_id}/artifacts/{filename}", tags=["jobs"])
    async def get_artifact(job_id: str, filename: str) -> Response:
        row = job_db.get(job_id)
        if not row:
            raise APIError(404, "JOB_NOT_FOUND", "No job exists with this job_id")
        result = json.loads(row["result_json"]) if row["result_json"] else None
        if row["status"] != "succeeded" or not result or filename not in result.get("files", {}):
            raise APIError(404, "ARTIFACT_NOT_FOUND", "The requested artifact is unavailable")
        try:
            content = object_storage.get_bytes(f"{job_id}/{filename}")
        except Exception as exc:
            logger.exception("Artifact retrieval failed", extra={"job_id": job_id, "correlation_id": row["correlation_id"]})
            raise APIError(503, "ARTIFACT_STORAGE_UNAVAILABLE", "Artifact storage is temporarily unavailable", True) from exc
        media_type = "video/mp4" if filename.endswith(".mp4") else "application/x-subrip" if filename.endswith(".srt") else "application/json"
        return Response(content=content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    return app


app = create_app()
