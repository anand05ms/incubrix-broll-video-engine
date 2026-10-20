import json
import logging


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "job_id": getattr(record, "job_id", None),
            "correlation_id": getattr(record, "correlation_id", None),
        }
        for field in (
            "scene_id",
            "visual_status",
            "visual_type",
            "visual_kind",
            "visual_path",
            "segment_path",
            "duration_seconds",
            "image_width",
            "image_height",
            "image_mode",
            "has_visible_pixels",
            "fallback_reason",
            "generated_queries",
            "providers_attempted",
            "provider_attempts",
            "provider_host",
            "request_method",
            "http_status",
            "failure_type",
            "request_duration_ms",
            "search_duration_ms",
            "selected_provider",
            "provider",
            "candidate",
            "candidate_count",
            "best_score",
            "selected_candidate",
            "fallback_used",
            "failed_visual_path",
            "fallback_path",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
                "message": str(record.exc_info[1]),
            }
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("broll")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger
