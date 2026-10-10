"""Celery app. Broker + result backend on Redis. Hard time limit so a hung
sampling loop can't pin a worker forever (fires the timeout -> failed path)."""

from __future__ import annotations

from celery import Celery

from config import JOB_TIMEOUT_S, REDIS_URL

celery = Celery("diffusion", broker=REDIS_URL, backend=REDIS_URL)
celery.conf.update(
    task_time_limit=JOB_TIMEOUT_S,            # hard kill
    task_soft_time_limit=JOB_TIMEOUT_S - 15,  # soft raise -> graceful 'failed: timeout'
    worker_prefetch_multiplier=1,             # don't hoard tasks on a slow GPU worker
    task_acks_late=True,
)

import tasks  # noqa: E402,F401  (register the task)
