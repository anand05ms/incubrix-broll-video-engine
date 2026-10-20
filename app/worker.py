import json
import time

from app.config import Settings
from app.db import JobDatabase
from app.jobs import execute_job
from app.logging_config import configure_logging
from app.storage import create_storage


logger = configure_logging()


def main() -> None:
    settings = Settings.from_env()
    if settings.job_execution_mode != "external":
        raise RuntimeError("Set JOB_EXECUTION_MODE=external before starting the worker process")
    database = JobDatabase(settings.database_path)
    storage = create_storage(settings)
    database.initialize()
    database.requeue_interrupted()
    logger.info("Queue worker started")
    try:
        while True:
            row = database.claim_next()
            if row is None:
                time.sleep(settings.job_poll_interval)
                continue
            execute_job(
                row["id"],
                json.loads(row["request_json"]),
                row["upload_path"],
                row["correlation_id"],
                database,
                settings,
                storage,
            )
    except KeyboardInterrupt:
        logger.info("Queue worker stopping")
    finally:
        database.close()


if __name__ == "__main__":
    main()
