"""FastAPI surface.

  POST /generate      rate-limit + supersede + enqueue -> {job_id, status}
  GET  /jobs/{id}     resolve status: Redis (hot) -> DB (durable) -> 404
  GET  /images        this IP's created images (durable, from DB)
  GET  /files/{key}   serve a stored image (S3 stand-in)

Run behind a proxy with:  uvicorn main:app --proxy-headers --forwarded-allow-ips=<lb>
so request.client.host is the real client IP, not the load balancer.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import func, text

import metrics as m
import ratelimit as rl
import storage
from config import (DDIM_STEPS, JOB_TIMEOUT_S, MAX_LABELS, MAX_QUEUE_DEPTH,
                    SAMPLER, SERVICE_TIME_S, UP_DDIM_STEPS)
from db import (PENDING, STARTED, SUCCESS, TERMINAL, Image, Job, engine, init_db,
                session_scope)
from tasks import run_generation


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Cascading Diffusion Inference", lifespan=lifespan)

# /metrics. Uses route templates for labels ("/jobs/{job_id}", not each uuid), which
# is what keeps label cardinality from exploding one series per job.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    """Liveness: is this process up. Deliberately checks NO dependencies — a failing
    DB must not make the orchestrator kill an otherwise healthy API."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz() -> dict:
    """Readiness: can this process actually serve. Checks both hard dependencies;
    503 pulls it out of the load balancer without a restart."""
    deps = {}
    for name, probe in (("postgres", _ping_db), ("redis", rl.ping)):
        try:
            probe()
            deps[name] = "ok"
        except Exception as exc:
            deps[name] = f"{type(exc).__name__}: {exc}"[:96]
    if any(v != "ok" for v in deps.values()):
        raise HTTPException(503, deps)
    return {"status": "ready", **deps}


def _ping_db() -> None:
    with engine.connect() as c:
        c.execute(text("select 1"))


def client_ip(request: Request) -> str:
    """Server-observed identity for rate limiting. Never trust a client-supplied
    value. IPv6 is bucketed to its /64 (a user owns the whole /64)."""
    host = request.client.host if request.client else "unknown"
    try:
        ip = ipaddress.ip_address(host)
        if ip.version == 6:
            return str(ipaddress.ip_network(f"{host}/64", strict=False).network_address)
    except ValueError:
        pass
    return host


ADMISSION_LOCK = 0x0D1FF5  # arbitrary constant; one advisory lock for the admit path


def _queue_depth(s) -> int:
    """Jobs pending or started server-wide, within the current session/transaction.

    Derived state, not a counter: a Redis INCR/DECR pair is cheaper but leaks on
    every crash, and a worker killed mid-job would leave the counter high forever,
    refusing traffic permanently. `status` is indexed, so this is one cheap count.

    Bounded by `created_at`: rows stuck 'started' behind a dead worker would
    otherwise consume capacity forever. Two job timeouts is comfortably past when
    Celery's hard limit should have finalised anything alive. (A reaper is still the
    proper fix — this only stops the leak being fatal.)
    """
    cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=2 * JOB_TIMEOUT_S)
    return s.query(func.count(Job.id)).filter(
        Job.status.in_((PENDING, STARTED)), Job.created_at >= cutoff).scalar() or 0


def _admit(s) -> int:
    """Serialise the admission decision, then return the depth we admitted at.

    Counting and inserting must be ONE atomic step. Measured 2026-08-03: without the
    lock, 16 simultaneous requests all read depth < cap before any row committed and
    15 were admitted against a cap of 12 — a textbook time-of-check-to-time-of-use
    race, and a cap that leaks 25% under burst is not a cap.

    Postgres only. On the SQLite fallback the race cannot happen — one writer at a
    time is exactly what made the *other* read-check-write guard in this codebase
    look correct until we moved to Postgres (see JOURNAL.md 2026-08-01).
    """
    if s.bind.dialect.name == "postgresql":
        s.execute(text("select pg_advisory_xact_lock(:k)"), {"k": ADMISSION_LOCK})
    depth = _queue_depth(s)
    m.QUEUE_DEPTH.set(depth)
    if depth >= MAX_QUEUE_DEPTH:
        m.ADMISSION_REJECTED.inc()
        retry = max(1, (depth - MAX_QUEUE_DEPTH + 1) * SERVICE_TIME_S)
        raise HTTPException(503, detail={
            "error": "queue full", "queue_depth": depth,
            "max_queue_depth": MAX_QUEUE_DEPTH, "retry_after_s": retry,
        }, headers={"Retry-After": str(retry)})
    return depth


@app.post("/generate")
def generate(labels: list[int], seed: int = 42, ip: str = Depends(client_ip)) -> dict:
    if not labels or len(labels) > MAX_LABELS:
        raise HTTPException(400, f"labels must be 1..{MAX_LABELS} class ids")
    if any(l not in (0, 1) for l in labels):
        raise HTTPException(400, "labels must be 0 (dog) or 1 (cat)")

    if not rl.allow_request(ip):
        raise HTTPException(429, "rate limit: max jobs per window reached")

    job_id = uuid4().hex
    params = {"labels": labels, "seed": seed, "sampler": SAMPLER,
              "steps": DDIM_STEPS, "up_steps": UP_DDIM_STEPS}

    # Admission control bounds AGGREGATE load, which the per-IP limit above cannot.
    # Shedding early keeps latency bounded for everyone already admitted -- a fast
    # 503 beats a 200 that resolves in two minutes.
    # The count and the INSERT share ONE transaction on purpose: split apart they are
    # a check-then-act race, and the row must become visible to the next caller's
    # count before this lock is released. A 503 rolls the transaction back, so a
    # rejected request leaves no row behind.
    with session_scope() as s:
        depth = _admit(s)
        s.add(Job(id=job_id, created_by=ip, params=params, status=PENDING))

    # Latest-wins: supersede whatever this IP had running.
    old = rl.claim_slot(ip, job_id)
    if old and old != job_id:
        rl.mark_cancelled(old)               # fast cancel signal for the worker loop
        with session_scope() as s:           # durable truth (guarded write in worker still applies)
            prev = s.get(Job, old, with_for_update=True)  # lock: races the worker's _finish
            if prev and prev.status not in TERMINAL:
                prev.status, prev.reason = "cancelled", "superseded"
                # Terminal means finished: without this, the worker's _finish correctly
                # refuses the already-terminal row and every duration query silently
                # drops cancelled jobs.
                prev.finished_at = dt.datetime.utcnow()

    rl.set_status(job_id, PENDING)
    run_generation.delay(job_id, labels, seed, ip)
    # Publish the wait so clients can back off themselves instead of hot-polling.
    return {"job_id": job_id, "status": PENDING, "queue_position": depth,
            "est_wait_s": depth * SERVICE_TIME_S}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    cached = rl.get_status(job_id)           # hot path: one Redis GET
    if cached and cached not in TERMINAL:
        return {"job_id": job_id, "status": cached}

    with session_scope() as s:               # cold path: one indexed PK read
        job = s.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found or expired")
        out = {"job_id": job_id, "status": job.status, "reason": job.reason}
        if job.status == SUCCESS:
            out["images"] = [storage.url_for(img.s3_key) for img in job.images]
        return out


@app.get("/images")
def my_images(ip: str = Depends(client_ip)) -> dict:
    with session_scope() as s:
        imgs = s.query(Image).filter(Image.created_by == ip).order_by(
            Image.created_at.desc()).all()
        return {"images": [storage.url_for(img.s3_key) for img in imgs]}


@app.get("/files/{key:path}")
def serve_file(key: str):
    path = storage.path_for(key)
    if not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)
