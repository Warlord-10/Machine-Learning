"""Central config for the serving stack. All Python, all env-overridable."""

from __future__ import annotations

import os
from pathlib import Path

SERVING_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SERVING_DIR.parent

# --- infra ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
METRICS_PORT = int(os.getenv("METRICS_PORT", "9100"))  # worker's /metrics (api uses 8000)
DB_URL = os.getenv("DB_URL", f"sqlite:///{SERVING_DIR / 'jobs.db'}")

# Local folder that stands in for an S3 bucket. Swap `storage.py` for real S3 later.
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(SERVING_DIR / "bucket")))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# --- rate limiting / concurrency (per client IP) ---
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "3"))          # jobs admitted...
RATE_WINDOW_S = int(os.getenv("RATE_WINDOW_S", "60"))   # ...per this rolling window

# --- job lifecycle ---
JOB_TIMEOUT_S = int(os.getenv("JOB_TIMEOUT_S", "300"))  # Celery hard limit + active-slot TTL
TERMINAL_TTL_S = int(os.getenv("TERMINAL_TTL_S", str(15 * 60)))  # cache warmth after finish

# --- generation defaults ---
SAMPLER = os.getenv("SAMPLER", "ddim")
DDIM_STEPS = int(os.getenv("DDIM_STEPS", "25"))        # base model (from noise);
# 25 measured indistinguishable from 50; 15 corrupts eyes. See JOURNAL.md.
# Upsampler gets its own, deliberately smaller budget: it is 83% of wall time and a
# refiner, not a generator. 50/25/15 measured indistinguishable; 6 is the choice.
UP_DDIM_STEPS = int(os.getenv("UP_DDIM_STEPS", "6"))
MAX_LABELS = int(os.getenv("MAX_LABELS", "4"))     # cap batch size per request

# --- admission control (aggregate load, NOT per-client) ---
# The per-IP rate limit bounds ONE client. It does nothing about many compliant
# clients arriving together: the 2026-08-02 load test showed 8 in-limit clients
# producing 382s tail latency with every request answered 200 OK. A queue cap is
# what turns "accept everything and be slow" into "say no and stay fast".
MAX_QUEUE_DEPTH = int(os.getenv("MAX_QUEUE_DEPTH", "12"))
# Used only to estimate wait time for clients (Retry-After / est_wait_s). Set it
# from measured service time; it is advisory, never a control input.
SERVICE_TIME_S = int(os.getenv("SERVICE_TIME_S", "8"))
