"""Worker: packs queued jobs into batched cascade pipeline runs.

Each RabbitMQ message calls `run_image_pipeline`, which adds its job to an in-process
queue and waits. `batch_loop` takes jobs off that queue, packs them into one model
pass of up to BATCH_MAX_IMAGES images, and resolves each job when its batch is done.
The message is acked when its handler returns, and dead-lettered if it raises.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import logging
import sys
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import cascade_pipeline

import numpy as np  # noqa: E402

from cascade_pipeline import CascadeConfig, CascadePipeline, to_uint8  # noqa: E402
import storage  # noqa: E402
from config import (BATCH_MAX_IMAGES, BATCH_MAX_WAIT_MS, JOB_TIMEOUT_S, SAMPLER,  # noqa: E402
                    TERMINAL_TTL_S)
from db import FAILED, PENDING, STARTED, SUCCESS, Image, Job, session_scope  # noqa: E402
from ratelimit import r as redis  # noqa: E402


@cache
def get_pipeline() -> CascadePipeline:
    """Loaded once per worker process (two ~200MB ONNX sessions), never per job."""
    return CascadePipeline(CascadeConfig(sampler=SAMPLER))


@dataclass
class QueuedJob:
    job_id: str
    labels: list[int]
    created_by: str
    done: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


log = logging.getLogger(__name__)
_queue: asyncio.Queue[QueuedJob] = asyncio.Queue()


async def run_image_pipeline(payload: dict) -> list[int]:
    """RabbitMQ handler: queue the job for batching and wait for its image ids."""
    job_id = payload["job_id"]

    # Only a pending job runs: cancelled, in-progress or finished ones are skipped.
    if await get_status(job_id) != PENDING:
        return []

    job = QueuedJob(job_id, payload["labels"], payload["created_by"])
    await _queue.put(job)
    return await job.done


async def batch_loop() -> None:
    """Forever: collect a batch of jobs and run it. A job that doesn't fit in the
    current batch is carried over to start the next one."""
    carried_over: QueuedJob | None = None
    while True:
        batch = [carried_over or await _queue.get()]
        carried_over = None
        image_count = len(batch[0].labels)
        deadline = asyncio.get_running_loop().time() + BATCH_MAX_WAIT_MS / 1000

        while image_count < BATCH_MAX_IMAGES:
            remaining = deadline - asyncio.get_running_loop().time()
            try:
                job = await asyncio.wait_for(_queue.get(), timeout=max(remaining, 0))
            except TimeoutError:
                break  # waited long enough: run what we have
            if image_count + len(job.labels) > BATCH_MAX_IMAGES:
                carried_over = job  # never split a job across batches
                break
            batch.append(job)
            image_count += len(job.labels)

        try:
            await run_batch(batch)
        except Exception as error:  # e.g. DB down: never leave a handler waiting forever
            log.exception("batch of %d job(s) failed", len(batch))
            for job in batch:
                if not job.done.done():
                    job.done.set_exception(error)


async def run_batch(batch: list[QueuedJob]) -> None:
    """One model pass for every job in the batch, then store each job's images."""
    for job in batch:
        await set_status(job.job_id, STARTED)

    labels = [label for job in batch for label in job.labels]
    log.info("batch: %d job(s), %d image(s)", len(batch), len(labels))
    try:
        # Random seed per batch. CPU-bound ONNX runs in a thread so the event loop
        # keeps answering RabbitMQ heartbeats.
        result = await asyncio.to_thread(get_pipeline().generate, labels)
    except Exception as error:
        for job in batch:
            await fail(job, error)
        return

    start = 0
    for job in batch:
        images = result.images[start:start + len(job.labels)]
        start += len(job.labels)
        try:
            image_ids = await store_images(job.job_id, job.created_by, job.labels, images)
            await set_status(job.job_id, SUCCESS)
            job.done.set_result(image_ids)
        except Exception as error:
            await fail(job, error)


async def fail(job: QueuedJob, error: Exception) -> None:
    await set_status(job.job_id, FAILED, reason=f"{type(error).__name__}: {error}"[:128])
    job.done.set_exception(error)


async def get_status(job_id: str) -> str | None:
    """Redis first (cheap); fall back to the DB when the key is missing or expired."""
    cached = await redis.get(f"job:{job_id}")
    if cached:
        return cached.decode()
    async with session_scope() as s:
        job = await s.get(Job, job_id)
        return job.status if job else None


async def set_status(job_id: str, status: str, reason: str | None = None) -> None:
    """DB first (source of truth), then the Redis cache."""
    now = dt.datetime.utcnow()
    async with session_scope() as s:
        job = await s.get(Job, job_id)
        job.status, job.reason = status, reason
        if status == STARTED:
            job.started_at = now
        else:
            job.finished_at = now

    ttl = JOB_TIMEOUT_S if status == STARTED else TERMINAL_TTL_S
    await redis.set(f"job:{job_id}", status, ex=ttl)


async def store_images(job_id: str, created_by: str, labels: list[int],
                       images: np.ndarray) -> list[int]:
    """Save each generated image to storage and record it as an `Image` row."""
    rows = []
    for i, (label, image) in enumerate(zip(labels, images)):
        key = f"{job_id}/{i}_label{label}.png"
        buf = io.BytesIO()
        PILImage.fromarray(to_uint8(image)).save(buf, format="PNG")
        await asyncio.to_thread(storage.save, key, buf.getvalue())  # file/S3 write off the loop
        rows.append(Image(job_id=job_id, created_by=created_by, s3_key=key))

    async with session_scope() as s:
        s.add_all(rows)
        await s.flush()  # INSERT now so the ids are assigned
    return [row.id for row in rows]
