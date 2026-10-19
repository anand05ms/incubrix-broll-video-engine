from pathlib import Path
from typing import Protocol

from minio import Minio

from app.config import Settings


class ObjectStorage(Protocol):
    def ensure_ready(self) -> None: ...
    def put_file(self, key: str, file_path: Path, content_type: str) -> None: ...
    def get_bytes(self, key: str) -> bytes: ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def ensure_ready(self) -> None:
        if not self.root.is_dir():
            raise RuntimeError("Local storage directory is unavailable")

    def put_file(self, key: str, file_path: Path, content_type: str) -> None:
        destination = self.root.joinpath(*key.split("/")).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Invalid storage key")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file_path.read_bytes())

    def get_bytes(self, key: str) -> bytes:
        path = self.root.joinpath(*key.split("/")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Invalid storage key")
        return path.read_bytes()


class MinioStorage:
    def __init__(self, settings: Settings) -> None:
        if not settings.s3_access_key or not settings.s3_secret_key:
            raise ValueError("S3_ACCESS_KEY and S3_SECRET_KEY are required for MinIO storage")
        self.client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_secure,
        )
        self.bucket = settings.s3_bucket

    def ensure_ready(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def put_file(self, key: str, file_path: Path, content_type: str) -> None:
        self.client.fput_object(self.bucket, key, str(file_path), content_type=content_type)

    def get_bytes(self, key: str) -> bytes:
        response = self.client.get_object(self.bucket, key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()


def create_storage(settings: Settings) -> ObjectStorage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.data_dir / "objects")
    if settings.storage_backend == "minio":
        return MinioStorage(settings)
    raise ValueError(f"Unsupported STORAGE_BACKEND: {settings.storage_backend}")
