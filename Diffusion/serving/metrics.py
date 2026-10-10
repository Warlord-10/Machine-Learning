"""Prometheus metric definitions. Single source of truth for names and labels so
the api and worker processes can't drift apart.

Buckets are in SECONDS and tuned to a CPU-bound diffusion sampler (~35-45s per
image). Default prometheus_client buckets top out at 10s, which would collapse
every one of our jobs into +Inf and make p95 meaningless.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

STAGE_SECONDS = Histogram(
    "diffusion_stage_seconds",
    "Wall time per pipeline stage",
    ["stage"],  # base | condition | upsample | encode_store
    buckets=(0.05, 0.25, 1, 2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120, 300),
)

JOB_SECONDS = Histogram(
    "diffusion_job_seconds",
    "End-to-end task wall time (all stages + persistence)",
    buckets=(1, 5, 10, 20, 30, 45, 60, 90, 120, 300),
)

JOBS_TOTAL = Counter("diffusion_jobs_total", "Terminal job outcomes", ["status"])
IMAGES_TOTAL = Counter("diffusion_images_total", "Images generated and stored")
JOBS_INFLIGHT = Gauge("diffusion_jobs_inflight", "Jobs currently sampling on this worker")

# Admission control. REJECTED rising while QUEUE_DEPTH sits at the cap is the
# signal to add workers; rising with a shallow queue means the cap is too low.
QUEUE_DEPTH = Gauge("diffusion_queue_depth", "Jobs pending or started (server-wide)")
ADMISSION_REJECTED = Counter("diffusion_admission_rejected_total",
                             "Requests shed by admission control (queue full)")
