import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, URL, create_engine, event, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    request_hash: Mapped[str] = mapped_column(String, nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    upload_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String, nullable=False)
    progress: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    correlation_id: Mapped[str] = mapped_column(String, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_json: Mapped[str | None] = mapped_column(Text)


class JobDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.engine = create_engine(
            URL.create("sqlite", database=str(path)),
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        event.listen(self.engine, "connect", self._configure_connection)

    @staticmethod
    def _configure_connection(connection: Any, _: Any) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()

    @staticmethod
    def _as_dict(record: JobRecord) -> dict[str, Any]:
        return {column.name: getattr(record, column.name) for column in JobRecord.__table__.columns}

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def create(
        self,
        job_id: str,
        idempotency_key: str,
        request_hash: str,
        request_data: dict[str, Any],
        upload_path: str | None,
        correlation_id: str,
    ) -> None:
        now = utc_now()
        with Session(self.engine) as session:
            session.add(
                JobRecord(
                    id=job_id,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    request_json=json.dumps(request_data, ensure_ascii=False),
                    upload_path=upload_path,
                    status="queued",
                    progress=0,
                    created_at=now,
                    updated_at=now,
                    correlation_id=correlation_id,
                )
            )
            try:
                session.commit()
            except IntegrityError:
                session.rollback()
                raise

    def get(self, job_id: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            record = session.get(JobRecord, job_id)
            return self._as_dict(record) if record else None

    def by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with Session(self.engine) as session:
            record = session.scalar(select(JobRecord).where(JobRecord.idempotency_key == key))
            return self._as_dict(record) if record else None

    def claim_next(self) -> dict[str, Any] | None:
        """Atomically claim one queued job so independent worker processes do not duplicate it."""
        now = utc_now()
        next_id = (
            select(JobRecord.id)
            .where(JobRecord.status == "queued")
            .order_by(JobRecord.created_at, JobRecord.id)
            .limit(1)
            .scalar_subquery()
        )
        with Session(self.engine) as session:
            record = session.execute(
                update(JobRecord)
                .where(JobRecord.id == next_id, JobRecord.status == "queued")
                .values(status="running", progress=1, updated_at=now)
                .returning(JobRecord)
            ).scalar_one_or_none()
            session.commit()
            return self._as_dict(record) if record else None

    def requeue_interrupted(self) -> list[dict[str, Any]]:
        with Session(self.engine) as session:
            session.execute(
                update(JobRecord)
                .where(JobRecord.status == "running")
                .values(status="queued", updated_at=utc_now())
            )
            session.commit()
            records = session.scalars(select(JobRecord).where(JobRecord.status == "queued").order_by(JobRecord.created_at))
            return [self._as_dict(record) for record in records]

    def update(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> None:
        changes: dict[str, Any] = {"updated_at": utc_now()}
        if status is not None:
            changes["status"] = status
        if progress is not None:
            changes["progress"] = progress
        if result is not None:
            changes["result_json"] = json.dumps(result)
        if error is not None:
            changes["error_json"] = json.dumps(error)
        with Session(self.engine) as session:
            session.execute(update(JobRecord).where(JobRecord.id == job_id).values(**changes))
            session.commit()
