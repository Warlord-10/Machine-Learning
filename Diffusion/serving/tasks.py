"""The generation task: the only place the ONNX pipeline runs.

State-write contract (DB is the commit point, cache is best-effort after):
  start   -> DB started        -> cache started
  success -> S3 + DB success   -> cache success
  fail    -> DB failed+reason   -> cache failed
  cancel  -> DB cancelled       -> cache already 'cancelled' (set by the API)
All DB terminal writes are GUARDED (only from pending/started) so a job that
finishes a moment after being superseded can't overwrite the cancelled row.
"""

from __future__ import annotations

import datetime as dt
import io
import sys
from pathlib import Path

from celery.signals import worker_process_init
from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image as PILImage
from prometheus_client import start_http_server

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import cascade_pipeline

from cascade_pipeline import CascadePipeline, CascadeConfig, JobCancelled, to_uint8  # noqa: E402
import storage  # noqa: E402
import ratelimit as rl  # noqa: E402
import metrics as m  # noqa: E402
from celery_app import celery  # noqa: E402
from config import DDIM_STEPS, METRICS_PORT, SAMPLER, UP_DDIM_STEPS  # noqa: E402
from db import (CANCELLED, FAILED, PENDING, STARTED, SUCCESS, Image, Job,  # noqa: E402
                session_scope)

# Loaded once per worker process (200MB x2 ONNX sessions), never per task.
_PIPE: CascadePipeline | None = None


@worker_process_init.connect
def _load_models(**_):
    global _PIPE
    _PIPE = CascadePipeline(CascadeConfig(sampler=SAMPLER, ddim_steps=DDIM_STEPS,
                                     up_ddim_steps=UP_DDIM_STEPS))
    # ponytail: one exporter per child process, so this binds only at --concurrency=1.
    # For >1, switch to prometheus_client multiprocess mode (PROMETHEUS_MULTIPROC_DIR).
    start_http_server(METRICS_PORT)


def _finish(job_id: str, status: str, reason: str | None = None) -> None:
    """Guarded terminal DB write + cache update."""
    with session_scope() as s:
        # SELECT ... FOR UPDATE: the guard is read-check-write, so it needs the row
        # lock to hold against the API's concurrent supersede write.
        job = s.get(Job, job_id, with_for_update=True)
        if job and job.status not in (SUCCESS, FAILED, CANCELLED):  # guard
            job.status = status
            job.reason = reason
            job.finished_at = dt.datetime.utcnow()
    rl.set_status(job_id, status)
    m.JOBS_TOTAL.labels(status=status).inc()


@celery.task(name="run_generation")
def run_generation(job_id: str, labels: list[int], seed: int, ip: str) -> None:
    # If it was superseded before a worker even picked it up, stop now.
    if rl.is_cancelled(job_id):
        _finish(job_id, CANCELLED, "superseded")
        rl.release_slot(ip, job_id)
        return

    with session_scope() as s:
        job = s.get(Job, job_id, with_for_update=True)
        if job and job.status == PENDING:
            job.status = STARTED
            job.started_at = dt.datetime.utcnow()
    rl.set_status(job_id, STARTED)

    try:
        with m.JOBS_INFLIGHT.track_inprogress(), m.JOB_SECONDS.time():
            result = _PIPE.generate(
                labels=labels, seed=seed,
                should_cancel=lambda: rl.is_cancelled(job_id),  # polled between steps
                on_stage=lambda stage, secs: m.STAGE_SECONDS.labels(stage=stage).observe(secs),
            )
            # Persist each image: S3 first (idempotent key), then DB rows.
            with m.STAGE_SECONDS.labels(stage="encode_store").time(), session_scope() as s:
                for i, label in enumerate(labels):
                    key = f"{job_id}/{i}_label{label}.png"
                    buf = io.BytesIO()
                    PILImage.fromarray(to_uint8(result.final[i])).save(buf, format="PNG")
                    storage.save(key, buf.getvalue())
                    # Idempotent: a redelivered task must not fail on the unique
                    # constraint, it must no-op. Same job retried, never concurrent.
                    if not s.query(Image.id).filter_by(job_id=job_id, s3_key=key).first():
                        s.add(Image(job_id=job_id, created_by=ip, s3_key=key))
            m.IMAGES_TOTAL.inc(len(labels))
        _finish(job_id, SUCCESS, None)

    except JobCancelled:
        _finish(job_id, CANCELLED, "superseded")
    except SoftTimeLimitExceeded:
        _finish(job_id, FAILED, "timeout")
    except Exception as exc:  # any generation error is a durable failure
        _finish(job_id, FAILED, f"{type(exc).__name__}: {exc}"[:128])
    finally:
        rl.release_slot(ip, job_id)
