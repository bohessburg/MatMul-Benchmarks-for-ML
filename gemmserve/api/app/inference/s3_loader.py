from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


@dataclass(frozen=True)
class S3ArtifactRef:
    bucket: str
    key: str
    region: Optional[str] = None


class ArtifactDownloadError(RuntimeError):
    pass


def _default_s3_client(region: Optional[str]):
    # Retries help with transient network issues.
    cfg = Config(retries={"max_attempts": 5, "mode": "standard"})
    if region:
        return boto3.client("s3", region_name=region, config=cfg)
    return boto3.client("s3", config=cfg)


def _safe_filename(key: str) -> str:
    # Turn "mnist/mlp/v1/weights.npz" into "mnist_mlp_v1_weights.npz"
    return key.replace("/", "_")


def _read_text(path: pathlib.Path) -> Optional[str]:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None


def _write_text_atomic(path: pathlib.Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def ensure_s3_artifact_local(
    ref: S3ArtifactRef,
    cache_dir: Optional[str] = None,
    s3_client=None,
) -> pathlib.Path:
    """
    Ensures the S3 object is present locally. Uses ETag to avoid unnecessary downloads.
    Returns local file path.

    This uses AWS's default credential chain (env, shared config, IAM role, etc.).
    """
    cache_root = pathlib.Path(cache_dir or os.getenv("MODEL_CACHE_DIR", "/tmp/gemmserve"))
    cache_root.mkdir(parents=True, exist_ok=True)

    local_path = cache_root / _safe_filename(ref.key)
    etag_path = local_path.with_suffix(local_path.suffix + ".etag")

    s3 = s3_client or _default_s3_client(ref.region)

    # HEAD first to get ETag (and fail fast if object missing)
    try:
        head = s3.head_object(Bucket=ref.bucket, Key=ref.key)
    except ClientError as e:
        raise ArtifactDownloadError(f"Failed head_object s3://{ref.bucket}/{ref.key}: {e}") from e

    etag = head.get("ETag")
    # ETag is usually quoted, like: '"abc123..."'
    etag = etag.strip('"') if isinstance(etag, str) else None

    current_etag = _read_text(etag_path)
    if local_path.exists() and etag and current_etag == etag:
        return local_path  # cache hit

    # Download to temp then atomic rename to avoid partial files
    tmp_path = local_path.with_suffix(local_path.suffix + ".download")
    try:
        s3.download_file(str(ref.bucket), str(ref.key), str(tmp_path))
    except ClientError as e:
        raise ArtifactDownloadError(f"Failed download_file s3://{ref.bucket}/{ref.key}: {e}") from e

    tmp_path.replace(local_path)
    if etag:
        _write_text_atomic(etag_path, etag)

    return local_path


def load_model_ref_from_env() -> S3ArtifactRef:
    bucket = os.getenv("MODEL_S3_BUCKET")
    key = os.getenv("MODEL_S3_KEY")
    region = os.getenv("AWS_REGION")

    if not bucket or not key:
        raise ValueError("MODEL_S3_BUCKET and MODEL_S3_KEY must be set")

    return S3ArtifactRef(bucket=bucket, key=key, region=region)