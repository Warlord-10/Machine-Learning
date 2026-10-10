"""Per-IP sliding-window rate limit + latest-wins concurrency, and the job-status
cache the worker polls for cancellation. Pure Python (redis-py); no Lua.

Redis keyspaces
  rl:ip:{ip}     ZSET  request timestamps      -> sliding window (TTL = window)
  active:{ip}    STR   current job_id for ip   -> concurrency slot (TTL = job timeout)
  job:{job_id}   STR   status string           -> poll cache + cancel signal
"""

from __future__ import annotations

import time
import uuid

import redis

from config import (JOB_TIMEOUT_S, RATE_LIMIT, RATE_WINDOW_S, REDIS_URL,
                    TERMINAL_TTL_S)
from db import CANCELLED, TERMINAL

r = redis.Redis.from_url(REDIS_URL)


def ping() -> None:
    """Raise if Redis is unreachable. For the API's /readyz probe."""
    r.ping()


# --------------------------------------------------------------------------- #
# Sliding-window rate limit (ZSET log). Atomic via a MULTI pipeline.
# --------------------------------------------------------------------------- #
def allow_request(ip: str, limit: int = RATE_LIMIT, window_s: int = RATE_WINDOW_S) -> bool:
    key = f"rl:ip:{ip}"
    now_ms = int(time.time() * 1000)
    window_ms = window_s * 1000
    member = f"{now_ms}:{uuid.uuid4().hex}"  # unique so same-ms requests don't collide

    pipe = r.pipeline()  # transaction=True -> MULTI/EXEC, the 4 ops apply atomically
    pipe.zremrangebyscore(key, 0, now_ms - window_ms)  # drop entries outside the window
    pipe.zadd(key, {member: now_ms})
    pipe.zcard(key)
    pipe.pexpire(key, window_ms)
    _, _, count, _ = pipe.execute()

    if count > limit:
        r.zrem(key, member)  # roll back: only admitted requests occupy the window
        return False
    return True


# --------------------------------------------------------------------------- #
# Concurrency: latest-wins. New job supersedes the old in-flight one.
# --------------------------------------------------------------------------- #
def claim_slot(ip: str, new_job_id: str) -> str | None:
    """Atomically set this ip's active job to new_job_id and return the previous
    one (or None). SET ... GET EX is a single atomic op (Redis >= 6.2)."""
    old = r.set(f"active:{ip}", new_job_id, ex=JOB_TIMEOUT_S, get=True)
    return old.decode() if old else None


def release_slot(ip: str, job_id: str) -> None:
    """Clear the slot only if it still points at this job (avoid clobbering a
    newer job that already superseded us)."""
    if r.get(f"active:{ip}") == job_id.encode():
        r.delete(f"active:{ip}")


# --------------------------------------------------------------------------- #
# Job-status cache (== the cancel signal). Worker polls this each step.
# --------------------------------------------------------------------------- #
def set_status(job_id: str, status: str) -> None:
    """In-flight statuses have NO ttl (must survive the whole job so the cancel
    signal is readable); terminal statuses get the 15-min warmth ttl."""
    key = f"job:{job_id}"
    if status in TERMINAL:
        r.set(key, status, ex=TERMINAL_TTL_S)
    else:
        r.set(key, status)


def get_status(job_id: str) -> str | None:
    v = r.get(f"job:{job_id}")
    return v.decode() if v else None


def mark_cancelled(job_id: str) -> None:
    """Signal a superseded job to stop. Explicit value — never absence."""
    set_status(job_id, CANCELLED)


def is_cancelled(job_id: str) -> bool:
    # Stop ONLY on the explicit value. A missing key (eviction) means 'proceed',
    # so a Redis memory blip can never kill a live job.
    return get_status(job_id) == CANCELLED
