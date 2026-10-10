# Serving Architecture — Cascading Diffusion Inference

Everything we agreed on, in points. Backend is **Python only** (no Lua; sliding
window done with a redis-py MULTI pipeline).

## Flow
- **`POST /generate`** returns a `job_id` immediately and pushes the work to a
  **Celery** queue. Never blocks the HTTP connection on ~80s inference.
- A **Celery worker** runs the ONNX cascade (`cascade_pipeline.py`), loading the
  models **once per worker process**, never per task.
- On success: image bytes go to **S3** (a local folder stands in for it), then a
  row per image is written to the **DB**; on failure/cancel: a durable terminal
  row with a `reason`.

## Source of truth vs cache
- **DB is authoritative** for terminal state. Two tables:
  - `jobs`: `id, created_by(ip), params, status, reason, created_at, started_at, finished_at`
  - `images`: `id, job_id(fk), created_by, s3_key, created_at` (stores the **key**, not a URL)
- **Redis is a best-effort cache** in front of it. A `jobs` row is created
  `pending` at submit, so a job that dies still leaves a trace.
- **Status resolution (`GET /jobs/{id}`)** is server-side, one path:
  `Redis (hot) → DB PK read (cold) → 404`. The client never orchestrates Redis-then-DB.

## Job state machine
`pending → started → success | failed | cancelled`
- `failed` carries a `reason` (`timeout`, error class). `cancelled` (`superseded`)
  is a **separate** state so it doesn't pollute the failure rate.
- Terminal DB writes are **guarded** (`WHERE status IN (pending, started)`) so a
  job finishing just after being superseded can't overwrite the `cancelled` row.
- The guard is a **read-check-write**, so it is only atomic while the row is
  locked: all three sites take `SELECT ... FOR UPDATE` (`s.get(..., with_for_update=True)`
  in `tasks._finish`, `tasks.run_generation`, `main.generate`). Under SQLite this
  was free — the file lock serialised every writer. Under Postgres (READ COMMITTED)
  two transactions can both read `started`, both pass the guard, and the later
  commit wins. **The guard without the lock is decoration.**

## Persistence (Postgres)
- `DB_URL=postgresql+psycopg://…`, one Postgres container, `pgdata` volume.
  Nothing else changed: no raw SQL and no dialect-specific column types anywhere,
  so the SQLAlchemy models port as-is.
- Schema is still created by `Base.metadata.create_all` at API startup. That holds
  until the first *altered* column; after that it silently does nothing and you
  need migrations (Alembic).
- `params` is `JSON`, not `JSONB`. Fine while nothing queries inside it;
  `JSONB` when you want to index on `params->>'seed'`.

## Redis keyspaces
| Key | Type | Purpose | TTL |
|---|---|---|---|
| `rl:ip:{ip}` | ZSET | sliding-window request log | = window (60s) |
| `active:{ip}` | STR | current job for the IP (concurrency slot) | = job timeout |
| `job:{job_id}` | STR | poll status **and** cancel signal | none in-flight; 15 min terminal |

- In-flight `job:{id}` has **no TTL** — it must survive the whole job so the
  cancel signal stays readable. Terminal status gets a **15-min warmth TTL**;
  after that a poll falls through to the DB.

## Rate limiting & concurrency (per server-observed IP)
- Identity = **client IP read server-side** (run uvicorn with `--proxy-headers`
  + trusted `--forwarded-allow-ips`). Never a client-supplied token/hash. IPv6
  bucketed to `/64`.
- **Sliding window: 3 jobs / 60s** — ZSET log, atomic via a redis-py MULTI
  pipeline; a rejected request is rolled back so only admits occupy the window.
- **Concurrency: latest-wins.** A new job supersedes the IP's in-flight one:
  `active:{ip}` is swapped atomically (`SET ... GET EX`), the old job's cache is
  set to `cancelled`, and its DB row updated. So 3/min can be submitted but only
  the latest runs to completion.

## Cooperative cancellation (no process kill)
- The worker polls `job:{id}` **between denoising steps** in `cascade_pipeline._sample`
  (every step for DDIM, every 20 for DDPM) and raises `JobCancelled` when it reads
  the explicit value `cancelled`.
- Stops on the **explicit value only** — a missing key (Redis eviction) means
  "proceed", so a memory blip can never kill a live job.
- Latency bounded to one step (~1s); no `revoke(terminate=True)`, no model reload.

## Reliability
- Celery **hard time limit** (`task_time_limit`) is the primary "job died" signal
  → soft-timeout raises → `failed: timeout`.
- **(Planned, not implemented)** a backstop cron reaps rows stuck `pending/started`
  past `started_at + worst_case_runtime` — only for the case the whole worker/box
  died and Celery couldn't write the terminal state. Guarded update.

## Deployment (Docker)
- **One image, two roles.** `api` and `worker` are the same build; compose overrides
  `command` for Celery. A YAML anchor (`x-app`) shares their env/volumes so the two
  cannot drift — the classic failure is a worker running different config than the API.
- **Models are mounted, not baked** (`./models:/app/models:ro`). 400MB of `.onnx`
  would otherwise land in every layer push. Cost: the image is not self-contained,
  so the mount is a deploy-time requirement.
- **Non-root** (uid 10001). `/data` is created and chowned in the image so the named
  volume inherits that ownership on first mount; a pre-existing root-owned volume
  must be chowned by hand once.
- **Two volumes, two lifecycles:** `data` (generated PNGs, the S3 stand-in) and
  `pgdata`. Neither is a bind mount — deliberately, to keep the repo clean.
- `restart: unless-stopped` + healthchecks on api and worker. `start_period` on the
  worker is 60s because the two ONNX sessions load before it can answer anything.

## Observability
- **`/healthz` vs `/readyz` is the important split.** Liveness checks *nothing* —
  if a failing Postgres made `/healthz` fail, the orchestrator would restart a
  perfectly healthy API and turn a dependency blip into a restart loop. Readiness
  probes both dependencies and 503s, which removes the instance from the LB
  *without* killing it.
- Metric definitions live in one module (`metrics.py`) so api and worker cannot
  disagree on names or labels.
- **Histogram buckets are explicit and in seconds** (out to 300s). The
  `prometheus_client` defaults stop at 10s, which would file every ~40s job into
  `+Inf` and make every quantile meaningless. *Default buckets assume a web request;
  ML inference is not a web request.*
- HTTP metrics use **route templates** (`/jobs/{job_id}`), never raw paths — raw
  paths mean one time series per job id, which is how you kill a Prometheus.
- `cascade_pipeline.py` stays metrics-free: `generate()` takes an optional
  `on_stage(name, seconds)` callback and the serving layer passes a Prometheus
  observer. The pipeline remains runnable as a standalone script.
- Worker exposes its own `/metrics` on `:9100` via `start_http_server` in
  `worker_process_init`. **This binds once per child process** — correct only at
  `--concurrency=1`; beyond that, `prometheus_client` multiprocess mode.

## Design patterns in use (and the reason, not the name)
| Pattern | Where | Why it's here |
|---|---|---|
| Async job queue / worker pool | Celery + Redis | inference is ~40-110s; never hold an HTTP connection open for it |
| Database-as-source-of-truth, cache-aside | `db.py` + `ratelimit.py` | Redis can be evicted at any moment; terminal state cannot be lost |
| Guarded state transition (compare-and-set) | `_finish`, supersede | concurrent writers to one job row; wrong order must lose |
| Pessimistic row lock | `with_for_update=True` | makes the guard actually atomic on Postgres |
| Cooperative cancellation | poll between denoise steps | killing a process would mean reloading 400MB of models |
| Repository / adapter seam | `storage.py` | swap local folder → boto3 without touching callers |
| Fail-open on cache, fail-closed on trust | cancel signal vs rate limit | missing Redis key means "proceed"; missing auth means "reject" |
| Sidecar-free in-process exporter | `metrics.py` | one fewer moving part at this scale; has a known ceiling (concurrency 1) |
| Config-as-env, single module | `config.py` | one place to see every knob; no scattered `os.getenv` |
| Callback seam over dependency | `on_stage` | keeps the pipeline importable without Prometheus installed |

## What actually destroys this at scale
Ordered by how likely it is to bite first. Each has the mitigation, not just the scare.

| # | What breaks | Why | Mitigation |
|---|---|---|---|
| 1 | **Throughput ceiling ~8 images/min** (7.6s/image; was 1.57/min at 50/50 steps before the 2026-08-02 step reduction — **load test not yet re-run at the new settings**) | CPU-only ONNX, `--concurrency=1` | GPU execution provider next. Step reduction is spent: base 25 / upsample 6 already sit one notch above visible degradation |
| 1b | ~~No global admission control~~ **FIXED 2026-08-03.** Cap on `pending`+`started` → `503` + `Retry-After`; admitted requests get `queue_position`/`est_wait_s`. Verified: 16-client burst → exactly 12 admitted / 4 shed, **p95 115.7s → 73.5s**. Queue wait is still ~5× service time when full — that is now a capacity decision, not an unbounded one | per-IP limiting bounds one client, never aggregate load | `MAX_QUEUE_DEPTH`. Depth is *derived* (indexed row count, windowed by `created_at`) not a counter, so it self-heals; count+INSERT share one txn under `pg_advisory_xact_lock` — split apart it leaked 15/12 under burst |
| 1c | **`active:{ip}` slot TTL expires mid-queue** — `ex=JOB_TIMEOUT_S` (300s) assumes queue wait ≈ 0. **Latent, not fixed:** faster service (7.6s) + the queue cap put max wait at ~71s, so it no longer reproduces; it returns if depth exceeds ~40 | TTL sized for runtime, not `queue_wait + runtime` | size the TTL for total time in system, or release the slot explicitly. **Do not mistake "stopped reproducing" for fixed** |
| 1d | **`--forwarded-allow-ips=*` lets any client spoof its rate-limit identity** with one header | needed for real client IPs behind a proxy, but `*` trusts everyone | narrow to the proxy's IP/CIDR |
| 2 | ~~Duplicate `images` rows on task retry~~ **FIXED 2026-08-02** | `task_acks_late=True` redelivers on worker crash; the insert wasn't idempotent | `UNIQUE (job_id, s3_key)` + skip-if-exists insert. Note `create_all` can't add this to an existing table — required a drop/recreate |
| 3 | **Batching is a trap** | 2 images cost ~2.9× one image, superlinear — memory-bandwidth bound. `MAX_LABELS=4` lets one request buy 4× the work | keep `MAX_LABELS` small; treat batch size as a cost multiplier, not a throughput win. Re-measure on GPU, where the answer inverts |
| 4 | **Unbounded storage growth** | nothing ever deletes from `/data/bucket`; every job adds PNGs forever | retention job (delete images whose `jobs.finished_at` is older than N days) + S3 lifecycle policy in real deployment |
| 5 | **Leaked `job:{id}` Redis keys** | in-flight status has **no TTL** by design (the cancel signal must survive the job). A worker that dies never writes terminal state, so the key never gets its 15-min TTL | the planned reaper cron; it fixes the stuck DB row and the Redis key together |
| 6 | **Rate limiting by IP collapses under NAT** | every user behind one CGNAT egress shares a 3/60s budget; conversely one attacker with many IPs is unlimited | API keys / auth as the identity, IP as a secondary signal only. IP-based limiting stops casual abuse, nothing more |
| 7 | **Postgres is a single point of failure** | one container, one volume, no backups, no replica | managed Postgres + PITR before anything real depends on it |
| 8 | **`create_all` is not migrations** | it creates missing tables and silently ignores changed columns — schema drift with no error | Alembic from the first column change onward |
| 9 | **`/files` is served by the API process** | blocking file IO on the event loop; also no CDN, and it scales with API replicas rather than with storage | real S3 + presigned URLs (`storage.py` is already the seam) |
| 10 | **Worker metrics exporter breaks at concurrency > 1** | `start_http_server` binds `:9100` per forked child | `prometheus_client` multiprocess mode, or a separate exporter process |
| 11 | **api exports worker metrics as `0`** | `main.py` imports `tasks.py`, so unlabeled worker metrics get registered in the API process too and report zero forever | query `{job="worker"}` or `sum()`. Misleading dashboards are an outage you can't see |
| 12 | ~~`finished_at` NULL on superseded jobs~~ **FIXED 2026-08-02** | the supersede write set `status`/`reason` only, then `_finish` correctly refused the terminal row | `finished_at` now set in the supersede branch |
| 14 | ~~`upsample_model.fp16.onnx` is corrupt~~ **MISDIAGNOSED — FIXED 2026-08-02.** The artifact is fine (hash-identical to a fresh conversion); **`ORT_ENABLE_ALL`'s layout optimizations miscompile it** on ort 1.28/aarch64 into a `Gather` over an empty tensor | the Python `SessionOptions` default is `ORT_ENABLE_ALL`, so the pipeline silently ran the one level that breaks | `_make_session` pins `ORT_ENABLE_EXTENDED`. **Never inherit an optimization-level default** — it isn't API contract and it changes between releases |
| 13 | **Grafana runs as anonymous admin** | `GF_AUTH_ANONYMOUS_ENABLED=true` — fine on a laptop, an open admin panel anywhere else | real auth before this leaves localhost |

## Not done yet (scope markers)
- Real S3 (swap the two functions in `storage.py` for boto3).
- Auth (API keys), Alembic migrations, retention/cleanup, CI, resource limits,
  the reaper cron, GPU provider (`CUDAExecutionProvider`).
- Grafana dashboards are unprovisioned on purpose — write the PromQL yourself.

## Files
| File | Role |
|---|---|
| `config.py` | env-overridable settings |
| `db.py` | SQLAlchemy `Job`/`Image`, states, session |
| `storage.py` | S3-stand-in (local folder) |
| `ratelimit.py` | sliding window, supersede, job-status/cancel cache |
| `celery_app.py` | Celery config (time limits) |
| `tasks.py` | the generation task (loads pipeline once/worker) |
| `main.py` | FastAPI endpoints, `/healthz`, `/readyz`, `/metrics` |
| `metrics.py` | Prometheus metric definitions (shared by api + worker) |
| `test_ratelimit.py` | sliding-window self-check (fake Redis) |
| `../Dockerfile` | one image for both roles, non-root |
| `../docker-compose.yml` | api, worker, redis, postgres, prometheus, grafana (+ adminer under `debug`) |
| `../observability/` | prometheus scrape config, grafana datasource provisioning |

## Run
```bash
docker compose up -d                              # whole stack
docker compose --profile debug up -d adminer      # + DB browser
```
| Endpoint | What |
|---|---|
| http://localhost:8000/docs | API |
| http://localhost:3000 | Grafana |
| http://localhost:9090 | Prometheus |
| http://localhost:8080 | Adminer (`debug` profile) |
| localhost:5432 | Postgres, for a desktop GUI |

Bare-metal (no Docker) still works — `config.py` falls back to SQLite + localhost Redis:
```bash
pip install -r requirements.txt        # + a running redis-server
celery -A celery_app.celery worker --loglevel=info &   # worker (loads models)
uvicorn main:app --proxy-headers --reload              # api (creates tables at startup)
```
