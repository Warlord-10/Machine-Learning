"""FastAPI surface.

  POST /generate      rate-limit + persist + enqueue -> {job_id, status}
  GET  /jobs/{id}     long poll: waits up to LONG_POLL_MAX_S for the job to finish

Run behind a proxy with:  uvicorn main:app --proxy-headers --forwarded-allow-ips=<lb>
so request.client.host is the real client IP, not the load balancer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import select, update

import ratelimit as rl
import storage
from config import (DDIM_STEPS, JOB_TIMEOUT_S, LONG_POLL_MAX_S, MAX_LABELS, SAMPLER,
                    TERMINAL_TTL_S, UP_DDIM_STEPS)
from db import (FAILED, PENDING, STARTED, SUCCESS, TERMINAL, Image, Job, engine, init_db,
                session_scope)
from rabbitmq import job_queue

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.5  # how often a held GET re-checks the Redis status


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await init_db()
    yield
    await job_queue.close()
    await engine.dispose()


app = FastAPI(title="Cascading Diffusion Inference", lifespan=lifespan)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/generate")
async def generate(request: Request, labels: list[int]) -> dict:
    # Validate input labels
    if not labels or len(labels) > MAX_LABELS:
        raise HTTPException(400, f"labels must be 1..{MAX_LABELS} class ids")
    if any(l not in (0, 1) for l in labels):
        raise HTTPException(400, "labels must be 0 (dog) or 1 (cat)")

    # System level rate limiting
    if not await rl.system_rate_limit():
        raise HTTPException(429, "Too many requests")

    # Core processing
    job_id = uuid4().hex
    params = {"labels": labels, "sampler": SAMPLER,
              "steps": DDIM_STEPS, "up_steps": UP_DDIM_STEPS}

    # Commit the row BEFORE publishing: the worker reads the job from the DB, so
    # publishing inside the transaction could hand it a job that isn't there yet.
    
    async with session_scope() as s:
        s.add(Job(id=job_id, created_by=request.client.host, params=params, status=PENDING))
    # Before publishing: written after, it could overwrite the worker's 'started'.
    await rl.r.set(f"job:{job_id}", PENDING, ex=JOB_TIMEOUT_S)

    try:
        await job_queue.publish({"job_id": job_id, "labels": labels,
                                 "created_by": request.client.host})
    except Exception:
        log.exception("enqueue failed for job %s", job_id)

        # Execute the command directly in the db (Saves network trips)
        async with session_scope() as s:
            await s.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(status=FAILED, reason="enqueue failed")
            )
        raise HTTPException(503, "job queue unavailable, retry shortly")

    return {"job_id": job_id, "status": PENDING}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str, wait: float = Query(LONG_POLL_MAX_S, ge=0, le=LONG_POLL_MAX_S)) -> dict:
    """Long poll: return as soon as the job finishes, or its current status after
    `wait` seconds. Clients call again until the status is final; wait=0 returns at once."""
    deadline = time.monotonic() + wait
    while True:
        response = await read_job(job_id)
        if response["status"] in TERMINAL or time.monotonic() >= deadline:
            return response
        await asyncio.sleep(POLL_INTERVAL_S)


async def read_job(job_id: str) -> dict:
    # One round trip for both keys:
    #   job_resp:{id}  full response of a finished job (written below)
    #   job:{id}       live status string ('pending' from the API, then the worker's)
    cached_resp, cached_status = await rl.r.mget(f"job_resp:{job_id}", f"job:{job_id}")
    if cached_resp:
        return json.loads(cached_resp)
    if cached_status and cached_status.decode() in (PENDING, STARTED):
        return {"job_id": job_id, "status": cached_status.decode(), "reason": None}

    async with session_scope() as s:
        job = await s.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found or expired")

        out = {"job_id": job_id, "status": job.status, "reason": job.reason}

        if job.status == SUCCESS:
            # Explicit query: async sessions can't lazy-load job.images.
            keys = await s.scalars(select(Image.s3_key).filter_by(job_id=job_id).order_by(Image.id))
            out["images"] = [storage.url_for(k) for k in keys]

    # A finished job never changes, so its response is safe to cache.
    if job.status in TERMINAL:
        await rl.r.set(f"job_resp:{job_id}", json.dumps(out), ex=TERMINAL_TTL_S)
    return out
