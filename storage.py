"""Persistence for prediction result images.

Writes to any S3-compatible store when S3_BUCKET is configured, otherwise to a
local directory. On a server use the bucket: anything on the instance disk dies
with the instance.

Storage failures never fail the request -- a prediction the user can see is
worth more than a training sample we can re-collect.
"""
import os
import uuid
from datetime import datetime, timezone

_s3_client = None


def _get_s3_client(config):
    """One client, built from config so any S3-compatible store works.

    Neon Object Storage, Cloudflare R2, DigitalOcean Spaces and MinIO all speak
    the S3 API but live at their own endpoints; real AWS needs no endpoint at
    all. Credentials left unset fall through to boto3's normal chain.
    """
    global _s3_client
    if _s3_client is None:
        import boto3  # imported lazily so the dependency is optional locally

        kwargs = {"region_name": config.AWS_REGION}
        if config.S3_ENDPOINT_URL:
            kwargs["endpoint_url"] = config.S3_ENDPOINT_URL
        if config.S3_ACCESS_KEY_ID and config.S3_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = config.S3_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = config.S3_SECRET_ACCESS_KEY

        _s3_client = boto3.client("s3", **kwargs)
    return _s3_client


def build_key(prefix, filename):
    """Date-partitioned key, which keeps S3 listings and lifecycle rules sane."""
    now = datetime.now(timezone.utc)
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    return f"{prefix}/{now:%Y/%m/%d}/{uuid.uuid4().hex}{ext}"


def load_upload(key, config):
    """Read a stored image back. Returns bytes, or None if it is gone.

    The mirror of save_upload: same key, whichever backend that key lives in.
    """
    if not key:
        return None

    try:
        if config.S3_BUCKET:
            client = _get_s3_client(config)
            return client.get_object(Bucket=config.S3_BUCKET, Key=key)["Body"].read()

        with open(os.path.join(config.LOCAL_UPLOAD_DIR, key), "rb") as fh:
            return fh.read()
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not read {key} ({type(exc).__name__}: {exc})")
        return None


def save_upload(raw_bytes, filename, config, prefix=None):
    """Persist an image. Returns a storage key, or None if not stored.

    `prefix` overrides the default key prefix, which is how failed predictions
    are filed separately from results without needing a second bucket.
    """
    if not config.STORE_UPLOADS or not raw_bytes:
        return None

    key = build_key(prefix or config.S3_PREFIX, filename)

    try:
        if config.S3_BUCKET:
            client = _get_s3_client(config)
            client.put_object(
                Bucket=config.S3_BUCKET,
                Key=key,
                Body=raw_bytes,
                ContentType="image/jpeg",
            )
            return key

        destination = os.path.join(config.LOCAL_UPLOAD_DIR, key)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as fh:
            fh.write(raw_bytes)
        return key
    except Exception as exc:  # noqa: BLE001 - storage must never break /predict
        print(f"WARNING: could not persist upload ({type(exc).__name__}: {exc})")
        return None
