import logging
from typing import Any

from app.callbacks import send_signed_callback
from app.config import Settings
from app.db import JobDatabase
from app.engine import EngineError, process_job
from app.storage import ObjectStorage


logger = logging.getLogger("broll")


def execute_job(
    job_id: str,
    request_data: dict[str, Any],
    upload_path: str | None,
    correlation_id: str,
    database: JobDatabase,
    settings: Settings,
    storage: ObjectStorage,
) -> None:
    database.update(job_id, status="running", progress=1)
    logger.info("Job started", extra={"job_id": job_id, "correlation_id": correlation_id})
    try:
        result = process_job(
            job_id,
            request_data,
            upload_path,
            database,
            settings,
            storage,
            lambda value: database.update(job_id, progress=value),
        )
        database.update(job_id, status="succeeded", progress=100, result=result)
        logger.info("Job completed", extra={"job_id": job_id, "correlation_id": correlation_id})
        _send_callback(
            job_id,
            "succeeded",
            result,
            None,
            request_data,
            correlation_id,
            settings,
        )
    except EngineError as exc:
        error = {
            "code": exc.code,
            "message": exc.message,
            "retryable": exc.retryable,
            "details": exc.details,
        }
        database.update(job_id, status="failed", error=error)
        logger.error("Job failed: %s", exc.message, extra={"job_id": job_id, "correlation_id": correlation_id})
        _send_callback(
            job_id,
            "failed",
            None,
            error,
            request_data,
            correlation_id,
            settings,
        )
    except Exception:
        logger.exception("Unexpected job failure", extra={"job_id": job_id, "correlation_id": correlation_id})
        error = {
            "code": "JOB_PROCESSING_FAILED",
            "message": "Unexpected job processing failure",
            "retryable": False,
        }
        database.update(job_id, status="failed", error=error)
        _send_callback(
            job_id,
            "failed",
            None,
            error,
            request_data,
            correlation_id,
            settings,
        )


def _send_callback(
    job_id: str,
    status: str,
    result: dict[str, Any] | None,
    error: dict[str, Any] | None,
    request_data: dict[str, Any],
    correlation_id: str,
    settings: Settings,
) -> None:
    callback_url = request_data.get("callback_url")
    if not callback_url:
        return
    try:
        send_signed_callback(
            callback_url,
            settings.callback_hmac_secret,
            {
                "job_id": job_id,
                "status": status,
                "progress": 100 if status == "succeeded" else None,
                "result": result,
                "error": error,
                "correlation_id": correlation_id,
            },
            correlation_id,
        )
    except Exception:
        logger.exception(
            "Signed callback delivery failed",
            extra={"job_id": job_id, "correlation_id": correlation_id},
        )
