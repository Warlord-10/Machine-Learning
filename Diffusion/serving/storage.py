"""Storage abstraction. Local folder stands in for S3 — swap the two functions
for boto3 `put_object` / `generate_presigned_url` and nothing else changes.

The DB stores the `key` only; callers mint a (short-lived, in real S3) URL from it
on read, which is what the image-url cache in `ratelimit`/`main` fronts.
"""

from __future__ import annotations

from pathlib import Path

from config import PUBLIC_BASE_URL, STORAGE_DIR


def save(key: str, data: bytes) -> None:
    """Write bytes at `key` (== boto3 put_object)."""
    path = STORAGE_DIR / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def url_for(key: str) -> str:
    """Return a fetchable URL for `key` (== generate_presigned_url).

    Local mimic: a path served by the FastAPI /files route. In real S3 this
    returns a presigned URL with its own expiry (match TERMINAL_TTL_S).
    """
    return f"{PUBLIC_BASE_URL}/files/{key}"


def path_for(key: str) -> Path:
    """Local-only: resolve a key to a file path for the /files route to serve."""
    return STORAGE_DIR / key
