"""S3 storage client for DroneDetect pipeline.

Handles listing, downloading, and uploading files to Scaleway Object Storage
(S3-compatible). Configured via environment variables.
"""

import logging
import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)

S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL", "https://s3.fr-par.scw.cloud")
S3_BUCKET = os.environ.get("S3_BUCKET", "mlops-data")
S3_ARTIFACTS_BUCKET = os.environ.get("S3_ARTIFACTS_BUCKET", "mldrone-artefacts")
S3_RAW_PREFIX = os.environ.get("S3_RAW_PREFIX", "raw/")
S3_FEATURES_PREFIX = os.environ.get("S3_FEATURES_PREFIX", "features/")
S3_MODELS_PREFIX = os.environ.get("S3_MODELS_PREFIX", "models/")
S3_SPLIT_PREFIX = os.environ.get("S3_SPLIT_PREFIX", "split/")
S3_REGION = os.environ.get("S3_REGION", "fr-par")


def get_s3_client():
    """Create a configured boto3 S3 client for Scaleway."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        region_name=S3_REGION,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=BotoConfig(
            max_pool_connections=10,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


def list_dat_files(client=None) -> list[str]:
    """List all .dat file keys under the raw prefix, sorted deterministically.

    Returns keys sorted alphabetically to guarantee reproducible ordering
    regardless of S3 listing pagination order.
    """
    if client is None:
        client = get_s3_client()

    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_RAW_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".dat"):
                keys.append(key)

    keys.sort()
    logger.info("Found %d .dat files on S3", len(keys))
    return keys


def download_file(key: str, dest_path: Path, client=None) -> Path:
    """Download a single S3 object to a local path.

    Uses multipart download automatically via boto3's transfer manager.
    """
    if client is None:
        client = get_s3_client()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Downloading s3://%s/%s -> %s", S3_BUCKET, key, dest_path)
    client.download_file(S3_BUCKET, key, str(dest_path))
    return dest_path


def upload_file(
    local_path: Path, key: str, client=None, bucket: str = S3_ARTIFACTS_BUCKET
) -> None:
    """Upload a local file to S3.

    Uses multipart upload automatically for files > 8 MB (boto3 default threshold).
    """
    if client is None:
        client = get_s3_client()

    config = TransferConfig(multipart_chunksize=100 * 1024 * 1024)
    logger.info("Uploading %s -> s3://%s/%s", local_path.name, bucket, key)
    client.upload_file(
        str(local_path),
        bucket,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
        Config=config,
    )


def upload_features(features_dir: Path, client=None) -> None:
    """Upload all feature files (.npz and .npy) from features_dir to S3."""
    if client is None:
        client = get_s3_client()

    for pattern in ("*.npz", "*.npy", "*.json"):
        for feat_file in sorted(features_dir.glob(pattern)):
            key = S3_FEATURES_PREFIX + feat_file.name
            upload_file(feat_file, key, client=client, bucket=S3_ARTIFACTS_BUCKET)


def upload_models(models_dir: Path, client=None) -> None:
    """Upload trained model files (.pkl and .pth) from models_dir to S3."""
    if client is None:
        client = get_s3_client()

    for pattern in ("*.pkl", "*.pth", "*.csv", "*.json"):
        for model_file in sorted(models_dir.glob(pattern)):
            key = S3_MODELS_PREFIX + model_file.name
            upload_file(model_file, key, client=client, bucket=S3_ARTIFACTS_BUCKET)


def upload_split(split_path: Path, client=None) -> None:
    """Upload a split indices file to S3."""
    if client is None:
        client = get_s3_client()

    key = S3_SPLIT_PREFIX + split_path.name
    upload_file(split_path, key, client=client, bucket=S3_ARTIFACTS_BUCKET)
