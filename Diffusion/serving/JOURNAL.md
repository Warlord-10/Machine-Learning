# Cascading Diffusion — Engineering Journal

Everything about the two diffusion models in this project: theory, architecture
choices, ONNX conversion, serving, deployment, and the dated log of what was tried
and what the numbers said.

Two kinds of content, deliberately kept apart:
- **Parts 1–5 are reference** — stable, look-things-up material. Edit in place when
  it becomes wrong.
- **Part 7 is an append-only log** — never rewritten, because architecture docs lose
  their history and the history is where the learning lives.

`ARCHITECTURE.md` (same folder) is the current-state design of the serving stack,
its design patterns, and what breaks at scale.

> **TL;DR of the pipeline:** a cheap 64×64 base model draws the global structure
> of a dog/cat, then a conditional super-resolution model upscales it to 256×256.
> Both are trained as noise-predicting U-Nets and exported to ONNX for inference.

---

## 1. The two models

| | **Base** (`diffusion-multiclass.ipynb`) | **Upsampler** (`cascading-diffusion-2.ipynb`) |
|---|---|---|
| Job | noise → 64×64 image | (64×64 guide + noise) → 256×256 image |
| Resolution | 64 | 256 |
| Timesteps `T` | 1000 | 400 |
| U-Net width `dim` | 64 | 96 |
| Depth (channel mult) | (1, 2, 4, 8) | (1, 2, 2, 4) |
| Attention at res | 16, 8 | 32 (+ bottleneck) |
| Inputs | image, time, label | image**(6ch)**, time, blur, label |
| Epochs / batch | 250 / 64 | 100 / 20 |
| Optimizer | Adam 1e-4, CosineDecay→0.1 | Adam 1e-4 |
| Weights | `weights/multiclass-new.weights.h5` | `weights/upsampler_checkpoint.weights.h5` |

Dataset: AFHQ animal faces, two classes — `0 = dog`, `1 = cat`. Images are
normalized from `[0, 255]` to **`[-1, 1]`** (`img/127.5 - 1`) so the data matches
the mean-0 Gaussian the diffusion process assumes.

---

## 2. Diffusion theory (the parts that drive the code)

### Forward process — add noise
We destroy an image over `T` steps by mixing it with Gaussian noise. The closed
form (so we can jump to any step `t` directly) is:

```
x_t = sqrt(alpha_bar_t) · x_0 + sqrt(1 - alpha_bar_t) · noise
```

where `alpha_bar_t` is the cumulative product of `alpha = 1 - beta`. In code this
is `forward_noise()`. `beta_t` is the "how much noise to add at step t" schedule.

### Why the **cosine** noise schedule (not linear)
Both models use `cosine_noise_schedule()` instead of the original linear one.

- A **linear** schedule (`beta` from 1e-4 → 0.02) destroys almost all signal in
  the last ~20% of steps — those steps teach the model almost nothing, and at
  higher resolution it hurts quality.
- The **cosine** schedule (Nichol & Dhariwal, *Improved DDPM*) keeps more signal
  through the middle of the trajectory, so every timestep is informative.

The formula: `f(t) = cos²(((t/T + s)/(1 + s)) · π/2)`, `alpha_bar_t = f(t)/f(0)`,
`beta_t = 1 − alpha_bar_t / alpha_bar_{t-1}`, clipped to `[0, 0.999]`.
`s = 0.008` is a small offset so `beta` isn't exactly 0 at the start.

> ⚠️ **Consequence used everywhere in inference:** cosine gives `alpha_bar[0] == 1`.
> A `t = 0` reverse step divides by `sqrt(1 - alpha_bar_0) = 0` → NaN. So every
> sampling loop stops at `t = 1`.

### Why predict the **noise** (ε-prediction) with **MSE**
The U-Net's target is the noise that was added, not the clean image:
`loss = MSE(true_noise, predicted_noise)`. Noise is already unit-scaled and
zero-mean, which makes the regression target well-conditioned, and it's
mathematically equivalent to score matching. Training step: pick a random `t` per
image, noise the image, ask the model to recover the noise.

---

## 3. Architecture choices (why each layer is there)

Both models are U-Nets (down path → bottleneck → up path with skip connections).
Shared building blocks:

- **Sinusoidal time embedding** → `Dense → swish → Dense` (width `4·dim`).
  The scalar timestep `t` has to be fed to a conv net; sinusoidal features give it
  a smooth, high-frequency encoding (same trick as transformer positions).
- **Label embedding**: `Embedding(num_classes, 4·dim)` — turns the class id into a
  vector the same width as the time embedding so they can be combined.
- **Context = concat(time, label)** (upsampler also concatenates a **blur**
  embedding). This single context vector conditions every block.
- **Scale-shift (FiLM) modulation** inside each `ResNetBlock`:
  `h = h · (1 + gamma) + beta`, where `gamma, beta` are predicted from the context.
  This is how time/class/blur get injected at *every* resolution without
  concatenating channels everywhere — cheap and expressive.
- **GroupNorm (8 groups) + swish** instead of BatchNorm: diffusion uses tiny/uneven
  batches and per-image statistics, where BatchNorm is unstable.
- **Self-attention at low resolutions** (16/8 for base, 32 for upsampler): lets the
  model reason about global layout (two ears, symmetric eyes). Attention is O(N²)
  in pixels, so it's only applied where the feature map is small enough to afford.

### Base-specific
- Conv stem `kernel=7` for a large receptive field up front.
- Downsample with strided `Conv2D(k=4, s=2)`; upsample with `Conv2DTranspose`.
- Post-norm ResNet (`conv → norm → swish`).

### Upsampler-specific (the interesting one)
- **6-channel input**: the noisy 256×256 target is concatenated channel-wise with
  the 256×256 low-res guide. That's how the super-res model "sees" what to upscale.
- **`space_to_depth(block_size=2)` at the input, `depth_to_space` at the output**:
  rearranges each 2×2 spatial patch into channels, so the network runs internally
  at 128×128 with 4× channels. Quarters the spatial area → far cheaper convs and
  attention at 256, fully reversible.
- **Pre-norm ResNet** (`norm → swish → conv`) with **`ZeroConv`** (zero-initialized)
  on the second conv and the attention output. Zero-init means each residual branch
  starts as identity, so early training is stable (idea from ControlNet / progressive
  growing). Output is divided by `sqrt(2)` to keep variance from growing with depth.
- **Dropout 0.1** for a bit of regularization at the larger model size.
- **Blur conditioning** — see below.

### Why a **cascade** instead of one 256×256 model
Training a single 256 diffusion model from scratch is expensive and unstable. A
cascade splits the job: a small 64 model handles global structure cheaply, and a
conditional super-res model adds detail. This is Google's *Cascaded Diffusion
Models* recipe.

> **Measured consequence (2026-08-01):** the cascade makes *training* cheap but
> moves ~83% of *inference* cost into the upsampler. The design choice that made
> training feasible is the same one that now bounds serving throughput. See Part 7.

### Blur / noise-conditioning augmentation (upsampler robustness trick)
At inference the base model's 64×64 output is *not* a clean training image — it has
its own artifacts. If the upsampler only ever saw pristine low-res guides in
training, it would break on the base model's output (train/test mismatch). So
during training the low-res guide is deliberately degraded:

1. downsample the real image to 64 (nearest),
2. **50% of the time** blur it with a Gaussian kernel of random `sigma ∈ [0.1, 1.0]`,
3. upsample back to 256 (nearest),
4. **pass that `sigma` in as the `blur_input`** so the model knows how degraded the
   guide is.

At inference we feed **`blur = 0`** (tell it the guide is clean). This is the same
principle as *conditioning augmentation* in the cascaded-diffusion paper.

---

## 4. Inference / sampling

Reverse ("denoise") process — start from pure noise and walk back to a clean
image. Two samplers are implemented (`config.sampler`):

**DDPM** (`Schedule.ddpm_step`) — ancestral sampling, walks every step `t = T-1 → 1`:

```
mean   = 1/sqrt(alpha_t) · (x_t − (1-alpha_t)/sqrt(1-alpha_bar_t) · pred_noise)
x_{t-1} = mean + sqrt(beta_t) · z        (z ~ N(0,1); no z added on the last step)
```

High quality, but needs the full `T` model calls (1000 / 400).

**DDIM** (`Schedule.ddim_step`, Song et al. 2021) — the **production default**.
Predicts `x_0` from the noise estimate and jumps along a short *subsequence* of
timesteps (default 50) instead of all `T`:

```
x0      = (x_t − sqrt(1-alpha_bar_t) · pred_noise) / sqrt(alpha_bar_t)   (clipped to [-1,1])
x_{t'}  = sqrt(alpha_bar_t') · x0 + sqrt(1-alpha_bar_t' − σ²) · pred_noise + σ·z
σ       = eta · sqrt((1-alpha_bar_t')/(1-alpha_bar_t)) · sqrt(1 − alpha_bar_t/alpha_bar_t')
```

`eta=0` → deterministic (same seed ⇒ same image), the usual production setting;
`eta→1` recovers DDPM-level stochasticity. ~20× fewer model calls at near-identical
quality — this is why real image services sample with DDIM (or DPM-Solver), not DDPM.
Its last hop targets the clean image (`alpha_bar=1`) directly, sidestepping the
`t=0` division-by-zero.

Full cascade in `cascade_pipeline.py`:

1. **Stage 1** — sample the base model from noise → 64×64 image.
2. **Bridge** — bicubic-resize 64→256 to build the conditioning guide.
3. **Stage 2** — sample the upsampler, conditioning on that guide with `blur=0`, → 256×256.

> **Known train/inference subtlety to revisit:** training built the guide with
> **nearest** resize; the pipeline bridges with **bicubic**. The blur augmentation
> makes the model somewhat robust to this, but matching them (or picking the
> resize that looks best) is a cheap thing to tune.

### Running it
Configuration lives in `CascadeConfig` / `main()` — no CLI args:

```bash
python3 cascade_pipeline.py          # writes cascade_output/*.png
```

Or as a library — pick a sampler and batch as many labels as you like:

```python
from pathlib import Path
from cascade_pipeline import CascadePipeline, CascadeConfig

cfg = CascadeConfig(sampler="ddim", ddim_steps=50)   # or sampler="ddpm"
pipe = CascadePipeline(cfg)
pipe.generate(labels=[0, 1, 0, 1], seed=42).save(Path("cascade_output"))
```

`generate()` also takes two optional callbacks used by the serving layer:
`should_cancel()` (polled between denoise steps) and `on_stage(name, seconds)`
(per-stage timing). Both default to `None`, so this module never imports Redis or
Prometheus and still runs as a standalone script.

Reference timing: DDIM 50-step, batch=2, on CPU ≈ **80s** measured bare-metal;
**113s** measured inside Docker with Prometheus/Grafana co-resident on the same
CPU. Both numbers are real — see Part 7 on why the environment is part of the
measurement. DDPM (full) is ~20× slower.

`test_cascade_pipeline.py` holds model-free self-checks (schedule math, the
no-noise-at-t=0 rule, resize range). Run `python3 test_cascade_pipeline.py`.

---

## 5. ONNX — conversion, parity, optimization

Conversion is a **one-off build step**, not a runtime pipeline: run it once per
trained checkpoint, commit the `.onnx`, and serving never touches TensorFlow again.
It lives in **`export_onnx.py`**. Output goes to **`models/`** — an FP32 `.onnx` per
model, plus an FP16 copy when `EXPORT_FP16=True`.

```bash
python3 export_onnx.py     # -> models/{base,upsample}_model.onnx  (+ .fp16.onnx)
```

> Note: the upsampler's FP32 export is slow in tf2onnx (~several minutes for the
> 256×256 graph) — the full run takes longer than the base model alone. FP16 gives
> ~2× smaller files (base: 220MB → 110MB, `max|Δ|` vs FP32 ≈ 5.9e-3, harmless since
> DDPM/DDIM self-correct).

### Export — `export()`
Uses **`tf2onnx.convert.from_function`**. The Keras model is wrapped in a
`@tf.function` with an explicit `input_signature` (one `tf.TensorSpec` per input),
then converted.

| Param | Value | What it does |
|---|---|---|
| `opset` | 18 | ONNX operator-set version. Higher = newer ops; must be supported by your onnxruntime. 18 is a safe modern default. |
| `input_signature` | TensorSpecs, batch=`None` | Declares each input's name, dtype, shape. Leading axis is `None` → **dynamic batch**. |
| `output_path` | `models/*.onnx` | Where the graph is written. |

### ✅ Dynamic batch (was a bug, now fixed)
The original notebook built each spec as `shape = [1] + list(inp.shape[1:])`,
hardcoding batch to `1` — so `session.run` with 2 images raised `INVALID_ARGUMENT`.
`export_onnx.py` uses `None` for the batch dim instead:

```python
shape = [None] + list(inp.shape[1:])   # None -> dynamic batch in the ONNX graph
```

Verified: inputs now export as `['unk__…', 64, 64, 3]` and any batch size runs.
The parity check deliberately feeds **batch=2** so a regression back to a static
batch dim would fail the export.

> **Lesson (worth keeping):** the batch axis of an exported model is a real design
> decision, not a detail. Static-batch exports are fine for fixed-throughput serving
> and let some accelerators optimize harder; dynamic-batch is the flexible default.
> Always check `session.get_inputs()[i].shape` after export — don't assume.

### Parity check (the non-negotiable step)
Right after export, run both the ONNX and the source Keras model on the same input
and assert they agree: `max|onnx − keras| < 1e-3` (ours: **1.9e-4**). **Never trust
an export you haven't parity-checked** — a silent op mismatch produces a
plausible-but-wrong model.

### Graph optimization (applied automatically at load — not a build step)
We do **not** serialize an optimized graph offline. onnxruntime runs these
optimizations every time you create an `InferenceSession`, so the pipeline gets them
for free. You'd only pre-serialize (`SessionOptions.optimized_model_filepath`) to
shave session startup time — not worth it here.

| `GraphOptimizationLevel` | What it does |
|---|---|
| `ORT_DISABLE_ALL` | none |
| `ORT_ENABLE_BASIC` | constant folding, redundant-node elimination |
| `ORT_ENABLE_EXTENDED` | + fused ops (conv+activation, etc.) — **what we pin** |
| `ORT_ENABLE_ALL` | + layout optimizations (provider-specific; **miscompiles our fp16 upsampler** on ort 1.28/aarch64) |

> ⚠️ **The Python default is `ORT_ENABLE_ALL`, not EXTENDED** (an earlier version of
> this doc said EXTENDED — wrong, and it hid a real bug for a week).
> `_make_session` now pins EXTENDED explicitly. Don't inherit this default: it isn't
> part of ORT's API contract, and ALL trades correctness for layout passes that buy
> nothing here. See the 2026-08-02 entry in Part 7.

### FP16 — `convert_fp16()`
`onnxruntime.transformers.float16.convert_float_to_float16` casts weights/compute to
half precision (~2× smaller, faster on supporting hardware).

| Param | Value | What it does |
|---|---|---|
| `keep_io_types` | `True` | keeps model **inputs/outputs float32** (weights/interior become fp16), so callers don't change. |

FP16 is lossy, so its parity tolerance is looser (`< 0.05`). That's harmless here —
the DDPM loop is self-correcting over hundreds of steps.

> **Exported but unused.** `models/*.fp16.onnx` exist; `cascade_pipeline.py` still
> defaults to the FP32 paths (`CascadeConfig.base_model` / `.up_model`) and the paths
> are not env-configurable. See Part 8.

### Other params worth knowing (not yet used here)
- **Dynamic batch** — the `None` fix above; do this before any real batching.
- **INT8 dynamic quantization** — `onnxruntime.quantization.quantize_dynamic` for a
  further ~2× shrink; needs an accuracy check (diffusion can be sensitive).
- **Execution providers** — `CPUExecutionProvider`, `CUDAExecutionProvider`,
  `CoreMLExecutionProvider` (Apple silicon). The pipeline tries its configured
  providers and falls back to CPU on failure.
  ⚠️ **CoreML + dynamic batch don't mix:** with the dynamic-batch export, CoreML
  compiles but fails at runtime (`Shape error` / prediction error) — it effectively
  wants static shapes. So CoreML is **off by default**; use CUDA on a server, CPU
  locally. If you specifically want CoreML on the Mac, export a separate
  static-shape fp16 MLProgram just for it.
- **CoreML flags** — `ModelFormat=MLProgram`, `MLComputeUnits=ALL`,
  `RequireStaticInputShapes` (note: CoreML prefers static shapes, which trades
  against the dynamic-batch fix — pick per deployment target).

---

## 6. Where this sits in the MLE workflow

`train (notebooks) → export+optimize (export_onnx.py, one-off) → serve (cascade_pipeline.py) → deploy (serving/)`.

Current state: training ✅, dynamic-batch FP32 + FP16 ONNX export with parity ✅,
batched offline inference with DDPM **and** DDIM ✅, online serving ✅
(FastAPI + Celery + Redis + Postgres, containerised, Prometheus/Grafana metrics,
health probes). Not yet done: INT8 quantization, auth, real S3, GPU provider.

**Measured cost profile** (CPU). At the original 50/50 steps the **upsampler was 83%
of wall time** (94.6s of 113s for a 2-image batch) and the 64×64 base model 16% —
which is why all optimization was aimed at the upsampler.

**Current defaults: base 25 / upsample 6 DDIM steps, fp32** → **7.6s per image
end-to-end**, down from ~39s (~5.1×, 31 ONNX calls instead of 100). Quality verified
on held-out seeds; see Part 7 (2026-08-02).

**Precision options**, all measured on CPU:

| upsample model | size | upsample stage | e2e / image | drift vs fp32 |
|---|---|---|---|---|
| fp32 (default) | 207MB | 11.2s | 7.6s | reference |
| fp16 | 104MB | ~11s (no win) | — | 5.2e-3 (lossless) |
| **int8** (`UP_MODEL=…int8.onnx`) | **53MB** | **4.2s** | **4.7s** | 24.0 dB (grain) |

fp16 is a *size* optimization on CPU, not a speed one — ops up-convert to fp32 to
compute. **int8 is the real win: 2.7× on the dominant stage, 3.9× smaller, visually
near-identical** (the low PSNR is quantization grain, which the metric overstates —
see Part 9). Cumulative best: **39s → 4.7s, ~8.3×.**

Batching 2 images costs ~2.9× one image on CPU (superlinear, memory-bandwidth bound),
so batch size is a cost multiplier here, not a throughput win. Re-measure on GPU,
where this should invert.

### Endpoints
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

### Files
| File | Role |
|---|---|
| `scripts/diffusion-multiclass.ipynb`, `scripts/cascading-diffusion-2.ipynb` | training (base, upsampler) |
| `scripts/_base_arch.py`, `scripts/_up_arch.py` | model architectures (class names match saved weights) |
| `scripts/export_onnx.py` | Keras → dynamic-batch ONNX (FP32 + FP16), with parity checks |
| `cascade_pipeline.py` | ONNX-only inference: `CascadePipeline`, DDPM/DDIM samplers |
| `scripts/test_cascade_pipeline.py` | model-free self-checks (schedule, DDIM, resize) |
| `models/` | exported `.onnx` — FP32 + `.fp16.onnx`, all batch-agnostic |
| `weights/` | source Keras `.weights.h5` |
| `serving/` | the online serving stack — see `serving/ARCHITECTURE.md` |
| `Dockerfile`, `docker-compose.yml`, `observability/` | deployment |

---
---

# Part 7 — Engineering log (append-only)

`ARCHITECTURE.md` says what the system **is now**. This says **how we learned it** —
what was tried, what broke, what the numbers said. Never rewrite an entry; add a new
one that supersedes it.

Format per entry: **what changed → why → what it taught → evidence**. An entry with
no evidence line is a belief, not a finding, and is labelled as such.

---

## 2026-08-01 — Dockerised the stack

**Changed.** One `Dockerfile` for both roles (compose overrides `command` for
Celery), `docker-compose.yml` with api/worker/redis, `.dockerignore`, and the
inference deps (`onnxruntime`, `numpy`, `pillow`) added to `requirements.txt`.

**Why.** The serving stack ran only from a shell with hand-started processes. No
reproducible environment means no trustworthy benchmark.

**Learned.**
- `requirements.txt` claimed inference deps came "from the parent project" — there
  was no parent requirements file. The venv had them; the declared dependency set
  never did. **Containerising exposed a dependency lie that local dev hid.** This is
  the most common thing containers catch on first build.
- Models mounted (`./models:/app/models:ro`), not baked: 400MB of `.onnx` in an image
  is a slow push and a fat layer. Trade-off accepted knowingly — the image is no
  longer self-contained, so the mount becomes a deploy requirement.
- Flat imports (`import config`) forced `WORKDIR /app/serving`. Import style
  constrains container layout; worth knowing before designing either.

**Evidence.** Built clean; submitted a job; worker loaded both ONNX sessions on
`CPUExecutionProvider`; image served over `/files` (200, `image/png`, 141KB).

---

## 2026-08-01 — Where the data actually lives

**Learned.** After dockerising, images went to `/data/bucket/<job_id>/…` on a named
volume, and `jobs.db` beside it — **not** in the repo. On macOS the host mountpoint
Docker reports (`/var/lib/docker/volumes/…`) is inside the Docker Desktop VM and is
not openable from Finder. Decision: keep data in volumes (don't pollute the working
tree), and reach it with tooling rather than a bind mount.

**Learned (SQLite constraint).** SQLite has **no network protocol** — it's a file.
So no desktop GUI can reach it without the file being on the host. That constraint,
not preference, is what made a *containerised* browser (`sqlite-web`) the answer.
Read-only mount + `-r`, because a second writer would risk locking out the worker.

**Cost of getting it wrong.** First attempt used `-R` for read-only. `-R` is
rows-per-page; read-only is `-r`. The container crash-looped with an
`invalid integer value` error. **Lesson: verify a flag actually means what you
assumed — the container started "successfully" and only the logs told the truth.**

---

## 2026-08-01 — SQLite → Postgres, and the bug the migration exposed

**Changed.** Postgres 16 container + `pgdata` volume, `DB_URL` via compose,
`psycopg[binary]`, `sqlite-web` replaced by Adminer. Then the important part:
`SELECT ... FOR UPDATE` at the three read-check-write sites.

**Why.** SQLite over a shared volume works for one host, but a desktop GUI can't
reach it, and it wouldn't survive a second worker node.

**Learned — the headline of this session.**
> The state-machine guard (`if job.status not in TERMINAL`) was **only correct
> because SQLite serialised every writer on a file lock.** On Postgres under READ
> COMMITTED, the API's supersede write and the worker's `_finish` can both read
> `started`, both pass the guard, and the later commit wins — so a superseded job
> could persist as `success`.

**Migrating to a "better" database made a correctness bug reachable that the
"worse" one had been hiding.** Nothing in the ORM needed changing — no raw SQL, no
dialect-specific types — which made it tempting to call the migration done at the
config change. The concurrency semantics were the actual work.

**Learned (process).** My first race test **proved nothing and looked like a pass.**
Both jobs returned `success`; the timestamps showed job A finished 4s *before* job B
was submitted, so the two writes never overlapped. Sampling had run 34s that time,
not the 39-43s I'd sized the 38s delay against.
> A concurrency test that doesn't demonstrate the two operations overlapped is not
> evidence. Re-run with a 15s offset to force a genuine mid-flight collision.

**Evidence (second run, real collision).** Job A `cancelled/superseded`, stayed
cancelled after its worker unwound, wrote **0** `images` rows. Job B `success`.

**Found while verifying, still open.** A superseded job's `finished_at` is `NULL` —
the API's supersede write sets `status`/`reason` only, then `_finish` correctly
refuses the already-terminal row. Pre-existing, equally true on SQLite. It silently
drops cancelled jobs from every duration query.

**Reference.** Postgres has no built-in GUI; it ships `psql` only. Everything
graphical is a third-party client on 5432.

---

## 2026-08-01 — Observability, and the first real measurement

**Changed.** `metrics.py` (shared definitions), per-stage timing via a new optional
`on_stage(name, seconds)` callback on `generate()`, `/healthz` + `/readyz`,
Prometheus + Grafana containers, non-root uid 10001, `restart: unless-stopped`,
healthchecks, Postgres password to gitignored `.env`, compose YAML anchor.

### The measurement that changes the optimisation plan

Batch of 2 images, 50 DDIM steps:

| stage | seconds | share |
|---|---|---|
| base (64×64) | 18.67 | 16% |
| condition (bicubic) | 0.01 | ~0% |
| **upsample (256×256)** | **94.59** | **83%** |
| encode_store | 0.05 | ~0% |

**The upsampler is the entire cost.** Same 50 steps as the base model, over 16× the
pixels. Every optimisation must target *this* stage — halving base-model steps caps
out at an 8% win. Before measuring, "reduce DDIM steps" and "quantize the model"
looked like one undifferentiated lever; they're two, and only one of them matters.

**Batching looks like a trap on CPU.** 2 images ≈ 113s vs ~39s for one → **2.9×,
superlinear**. Consistent with memory-bandwidth-bound, not compute-bound.
*Labelled honestly: single sample, and Prometheus + Grafana were now competing for
the same CPU that the earlier single-image runs had to themselves.* Re-run both
under identical conditions before trusting the ratio.
> **A benchmark that changes its own environment isn't a benchmark.** Adding the
> observability needed to measure the system also perturbed the thing measured.

### Instrumentation lessons

- **Default histogram buckets stop at 10s.** Every ~40s job would land in `+Inf` and
  every quantile would be meaningless — metrics present, dashboards green, numbers
  worthless. Buckets are explicit, out to 300s. *Library defaults assume a web
  request; ML inference is not a web request.*
- **Label cardinality is a footgun.** Raw request paths would mint one time series
  per job UUID. `prometheus-fastapi-instrumentator` templates routes
  (`/jobs/{job_id}`) — that's the reason to prefer it over a hand-rolled middleware.
- **The api process exports worker metrics as `0` forever**, because `main.py`
  imports `tasks.py` and re-registers the unlabeled metrics. Query `{job="worker"}`
  or `sum()`. Noticed only because `diffusion_images_total` returned two series
  (`0` and `2`). Labeled counters don't have this problem — they emit no series
  until first use.
- **Liveness must not check dependencies.** If `/healthz` failed when Postgres
  blipped, the orchestrator would restart a healthy API and convert a dependency
  blip into a restart loop. `/readyz` probes dependencies and 503s instead —
  out of the load balancer, still alive.
- **Keep the model code free of serving concerns.** `cascade_pipeline.py` gets an
  `on_stage` callback; it never imports Prometheus and still runs as a script.

### Container lessons

- Going non-root broke writes to the **pre-existing** volume: `/data` is chowned in
  the image so a *fresh* named volume inherits uid 10001, but a volume already
  populated as root keeps root ownership. Needed a one-off
  `docker run --rm -v diffusion_data:/data alpine chown -R 10001:10001 /data`.
  **Image-level ownership only applies on first mount.**
- Worker `start_period` is 60s: it can't answer a healthcheck until two ONNX
  sessions are loaded. A healthcheck sized for a web app would flap forever.

### PromQL to start dashboards from
```promql
histogram_quantile(0.95, sum by (le,stage) (rate(diffusion_stage_seconds_bucket[5m])))
sum by (stage) (rate(diffusion_stage_seconds_sum[5m]))     # time share per stage
rate(diffusion_jobs_total[5m])                              # throughput by status
diffusion_jobs_inflight{job="worker"}
```

---

## 2026-08-02 — Load test: the queue is the system

**Changed.** Fixed both open bugs first, so the numbers would mean something:
`UNIQUE (job_id, s3_key)` on `images` + an idempotent insert (a redelivered task
must no-op, not fail on the constraint), and `finished_at` set in the supersede
branch. `create_all` **cannot** add a constraint to an existing table — dropped and
recreated both tables. That is the Alembic argument, demonstrated rather than argued.

**How the load test was possible at all.** Two things blocked it, and neither needed
a code change:
- Rate limit is 3/60s **per IP**, and all host traffic arrives as one IP.
- Worse, same-IP jobs **supersede each other by design** (latest-wins), so a
  single-identity test can never queue more than one job.

Uvicorn already runs with `--forwarded-allow-ips=*`, so sending a distinct
`X-Forwarded-For` per virtual client produces distinct rate-limit identities.
> **That is also a real hole:** with `*`, any client can spoof its own rate-limit
> identity by setting one header. Acceptable only behind a proxy that overwrites
> `X-Forwarded-For`. Narrow it to the proxy IP before exposing this.

**Result.** 10 jobs, 8 identities, fired concurrently. 10/10 success.

| metric | value |
|---|---|
| service time (sampling) | 35.8 – 53.6s (mean ~41s) |
| **queue wait** | **0.1 → 337.2s** |
| total latency avg / max | 198s / **382s** |
| throughput | **~1.57 jobs/min** (confirms the predicted ceiling) |

> **These load numbers are from the 50/50-step configuration.** The step reduction
> later that day cut service time to ~7.6s, so the throughput ceiling is now roughly
> **8 jobs/min** and every queue-wait figure below shrinks with it. The *shape* of the
> findings is unchanged — one worker, FIFO, no admission control — but **the load test
> must be re-run before any of these numbers are quoted again.** Left as measured
> rather than rewritten: the conclusions came from these runs.

**Learned — queue wait is 8× the work.** At saturation, service time stops being
the interesting number; latency is set by how many requests are ahead of you. No
sequential test can show this. p95 total latency was ~340s for 41s of actual work.

**Learned — a rate limiter is not capacity management.** Eight clients, each
comfortably *within* the per-IP limit, collectively buried the system for six
minutes and every one got an immediate `200 OK`. There is **no global admission
control**: no queue-depth cap, no backpressure, no "full, retry later". Per-IP
limiting stops one abusive client and does nothing about aggregate load.

### The bug that only load could find

`claim_slot` sets `active:{ip}` with `ex=JOB_TIMEOUT_S` (300s). **That TTL silently
assumes queue wait ≈ 0.**

Job `10.1.0.7` waited **337s** before starting — its concurrency slot expired at
300s, *while still queued*. For 37s that IP looked idle, so a new submission would
have found no old job to supersede, and the "only the latest job per IP runs"
invariant breaks: two jobs from one IP could run at once.

The slot TTL must cover `queue_wait + runtime`, or the slot should be released
explicitly rather than by expiry. **Not findable by code review or single-job
testing** — the bug requires a queue to exist. This is the argument for load testing
in one example.

---

## 2026-08-02 — The optimized model that had never been run

**Changed.** Made model paths env-overridable (`BASE_MODEL`, `UP_MODEL` in
`CascadeConfig`) so variants can be A/B'd without editing code. Then benchmarked
fp32 vs fp16 on the **upsample** model only — the only stage that can move the number.

**Result: there is no fp16 number, because `upsample_model.fp16.onnx` cannot run.**

```
Gather node 'Gather_token_662': indices element out of data bounds,
idx=0 must be within the inclusive range [0,-1]
```

Hypothesis "fp16 broke the dynamic batch axis" was **wrong** — it fails at batch 1,
2 and 4 alike. Probed all four artifacts:

| artifact | runs? |
|---|---|
| `base_model.onnx` | ✅ |
| `base_model.fp16.onnx` | ✅ finite output |
| `upsample_model.onnx` | ✅ batch 1/2/4 |
| **`upsample_model.fp16.onnx`** | ❌ **every batch size** |

**SUPERSEDED BY THE NEXT ENTRY.** The initial conclusion here — "the fp16 export is
corrupt, `convert_float_to_float16` cast a shape subgraph, re-export with an
`op_block_list`" — was **wrong**. The file is fine. Kept as written because the
wrong diagnosis is the instructive part; see below for how it collapsed.

**fp32 upsample baseline, uncontended:** 34.4s of a 40.8s single-image job (84%),
consistent with the 83% measured under Docker on 2026-08-01.

---

## 2026-08-02 — The model was never broken: ORT_ENABLE_ALL miscompiles it

**What actually happened.** Chasing the "corrupt fp16 export", the cheap fix was to
re-convert from the fp32 file with the current toolchain. The fresh file **ran fine**
— so, apparently, a tooling-version problem. Then the hash:

```
c9487f44a7c37e6c84a7d259aa1044a3  upsample_model.fp16.BROKEN.onnx
c9487f44a7c37e6c84a7d259aa1044a3  upsample_model.fp16.onnx   (freshly converted)
```

**Byte-identical.** The conversion is deterministic and the artifact was never
corrupt. The same bytes fail in the container and succeed on the host.

**Root cause.** It's the *graph optimizer*, not the file:

| optimization level | ort 1.28.0 / aarch64 Linux |
|---|---|
| `ORT_ENABLE_ALL` | **FAIL** — Gather over empty tensor |
| `ORT_ENABLE_EXTENDED` | OK |
| `ORT_ENABLE_BASIC` | OK |
| `ORT_DISABLE_ALL` | OK |

`ORT_ENABLE_ALL` is where **layout optimizations** live — the level this repo's own
notes already flagged as "can be provider-specific". On ort **1.28.0/aarch64** they
miscompile the fp16 upsampler into a `Gather` indexing an empty tensor. Host ort
**1.23.2/macOS** optimizes the identical bytes correctly, which is why it looked like
a corrupt artifact.

**And the documented default was wrong.** This repo stated the default level is
`ORT_ENABLE_EXTENDED`. In the Python API, `SessionOptions.graph_optimization_level`
defaults to **`ORT_ENABLE_ALL`** — so the pipeline was silently running the one level
that breaks, having "documented" itself as safe.

**Fix.** `_make_session` now pins `ORT_ENABLE_EXTENDED` explicitly instead of
inheriting a default that changes between releases. Every fusion that matters is at
EXTENDED; ALL adds layout passes worth ~nothing here and cost correctness.

### The lessons, which are better than the ones from the wrong diagnosis

> **1. "Same input, different result" means the environment is a variable.** Two ORT
> versions, two CPU architectures, one file. Everything blamed on the artifact was
> actually the runtime. **Hash the bytes before theorising about them** — one `md5sum`
> would have killed the wrong hypothesis 30 minutes earlier.

> **2. A fix that works is not a diagnosis.** Re-converting "fixed" it and would have
> shipped a confident, wrong story: *"my fp16 export was corrupt."* The only reason
> that didn't happen is the file size looked suspiciously identical. **A green result
> from a change you don't understand is a coincidence you haven't caught yet.**

> **3. Never inherit a default you depend on.** The default optimization level is not
> part of ORT's API contract; it changed, and silently selected the broken path.
> Pin correctness-relevant knobs explicitly.

> **4. Warnings are load-bearing.** `MergeShapeInfo ... source:{8} target:{0}` was
> printed at session creation every single time. It named the exact tensor that later
> failed, and it scrolled past as noise.

### And now the measurement it was blocking: fp16 buys size, not speed

With EXTENDED pinned, the A/B finally ran (1 image, 50 DDIM steps, alternating
order, n=3):

| variant | load | total (med) | base (med) | **upsample (med)** | file |
|---|---|---|---|---|---|
| fp32 | 1.4s | 37.6s | 5.8s | **32.1s** | 207MB |
| fp16 | 1.2s | 36.2s | 5.8s | **30.8s** | 104MB |

Per-run upsample times — fp32: 32.1 / 34.3 / 30.4; fp16: 31.5 / 26.6 / 30.8.

**Verdict: no measurable latency win.** 4% on the median against ~12% run-to-run
spread at n=3 is noise, not a speedup. The prediction was right and for the right
reason: on a **CPU** execution provider, fp16 ops are up-converted to fp32 to
compute, so you save weight memory and bandwidth but the arithmetic is unchanged.
`base_med` is identical (5.8s both) exactly as expected — only the upsampler was
swapped, which is a nice internal control on the harness.

**What fp16 actually bought:** 2× smaller artifact (207 → 104MB) and marginally
faster session creation. Those are **deploy-size and cold-start** wins — real, but
not the thing anyone reaches for fp16 hoping to get.

> The interesting claim isn't "fp16 is 2× faster" (it isn't, here). It's knowing
> **which resource a technique actually targets**, so you can predict where it pays:
> fp16 pays on hardware with native half-precision math (GPU tensor cores), and pays
> in bytes everywhere. Expect this result to **invert on GPU** — re-run it there.

**Still not measured:** INT8 dynamic quantization, which *does* change CPU
arithmetic and is the quantization worth trying on this hardware.

**Also corrected:** the earlier claim that the artifact "had never been executed" and
that parity was only run on the base model. `export_onnx.py:fp16_parity` **does** run
each fp16 model at batch=2 — but the timestamps (`upsample_model.onnx` 21:51 vs its
`.fp16` 23:00, an hour apart, while `process()` writes both together) show the fp16
was produced by a **side path that skipped the verified export flow**. The check
existed; the artifact bypassed it. On the host, where it was made, it passes.

---

## 2026-08-02 — 5.1x faster from step count alone (39s -> 7.6s)

**The winning lever was not a numeric format.** fp16 bought nothing on CPU and INT8
was ruled out on quality grounds. What worked: **fewer denoising steps**, aimed
per-stage.

**Blocker found first.** One `ddim_steps` fed *both* models, so the upsampler could
never be given fewer steps than the base — the single most valuable knob in the
system was not expressible. `_sample()` now takes an explicit `steps` argument and
`CascadeConfig` has separate `ddim_steps` (base) and `up_ddim_steps`, both
env-overridable (`DDIM_STEPS`, `UP_DDIM_STEPS`), and `up_steps` is recorded in each
job's `params` so any stored image can be traced to the budget that made it.

### Upsampler sweep (base pinned at 50)

| up_steps | PSNR vs 50-step ref | read |
|---|---|---|
| 25 | 39.6 dB | identical |
| 15 | 35.6 dB | imperceptible |
| 10 | 32.1 dB | slight |
| **6** | **27.7 dB** | visible softening of fine detail — **chosen** |
| 4 | 24.1 dB | clearly degraded |
| 2 | 16.8 dB | green mush; sampler breaks down |

Held-out check on 4 unseen seeds gave 15-vs-6 PSNR of 30.0/29.6/29.5/34.0 dB
(mean 30.8), matching 30.3/31.5/30.3/33.2 on the seeds used to pick the setting.
**The degradation is a property of the step cut, not of the seeds I'd been staring
at** — judging a choice on the images used to make it is circular, so the held-out
seeds were the real test.

### Base sweep (upsampler pinned at 6)

50 → 25 → 15 base steps. **25 is indistinguishable from 50** (mean 31.7 dB, same
subjects, same poses). 15 begins corrupting eyes.

### Final defaults and result

| | before | after |
|---|---|---|
| base steps | 50 | **25** |
| upsample steps | 50 | **6** |
| ONNX calls per image | 100 | **31** |
| end-to-end via the API | ~39s | **7.6s** |

**~5.1x, with no new artifacts, no quantization and no GPU.** Both knobs are env
vars, so quality tiers are configuration, not a fork. Precision stays **fp32** — fp16
is a size optimization here, not a speed one.

### Learned

> **"Reduce sampler steps" is not one optimization, it is one per stage.** Aimed at
> the base model it buys ~4s; aimed at the upsampler (83% of cost) it buys ~93s.
> Same technique, 20x difference in payoff, decided entirely by which stage you point
> it at. Measuring the stage split first is what made this obvious.

> **Latency is linear in steps** — ~2.1 s/step for the upsampler across 50 and 15,
> confirmed on an idle machine. Worth verifying rather than assuming: if there were a
> large fixed cost per stage, step reduction would have hit diminishing returns fast.

**"Are these images identical? Is sampling even running?"** — a fair challenge that
deserved evidence, not assertion. Instrumented `_predict_noise` to count real ONNX
calls: `up_ddim_steps` 15/6/2 produced exactly 15/6/2 upsample calls with base at 50.
The outputs differ on **99.9% of pixels** at low amplitude (mean |diff| 7.1/255) —
truly identical output would report `identical=True`. Low-amplitude-everywhere is the
signature of *changed texture*, not changed content.

**Prediction that was WRONG, kept deliberately.** I predicted cutting *base* steps
would change **what** the image is (different animal), since the base model
integrates from pure noise while the upsampler only refines fixed conditioning. It
didn't — same animals at 25 steps. DDIM at `eta=0` with `x0` clipping tracks the same
trajectory endpoint with half the hops. **The tidy "base = structure, upsample =
detail" story was too neat**; the honest version is that both stages are far more
robust to step reduction than the 50-step default implies, and the default was never
justified by measurement in the first place.

**Both current settings sit one notch above a cliff** (base 15 corrupts eyes,
upsample 2 is mush). Deliberate, but there is little headroom — worth a comment for
whoever lowers them next.

### Benchmark hygiene, learned the hard way (three times)

Every timing measured while anything else ran was worthless, and I re-learned it
repeatedly in one session:
- The 50-step upsample stage measured **94.2s and 207.2s** for identical work.
- A 6-step run appeared *slower* than a 15-step run, because I launched it while the
  sweep was still going — two inference containers on 8 cores.
- Even with the stack stopped, the **base stage measured 6.0s to 42.4s** while pinned
  at 50 steps, so the host was still not quiet.

> The invariant control saved this: base steps were fixed across the upsample sweep,
> so **base-stage variance is a direct readout of environmental noise.** Any run whose
> base time was off is a run whose other numbers are unusable. Build a constant into
> the experiment and it tells you when to distrust the rest of it.

Only two numbers here are trustworthy: **~2.1 s/step** (idle, alternating order) and
the **7.6s end-to-end** through the production path.

---

## 2026-08-02 — INT8: the win I predicted wouldn't happen

**Prediction, stated in advance: "a modest speedup at best, possibly none."** Wrong.
INT8 dynamic quantization of the upsampler is the second-biggest win of the day.

`quantize_dynamic(weight_type=QuantType.QUInt8)` on `upsample_model.onnx` — no
calibration set needed (that's *static* quantization). QUInt8 rather than QInt8
because int8 *activations* want VNNI-class instructions to be fast and can overflow
on some x86.

**Diagnostic first, which is what made the result predictable in hindsight:**

| graph | ops |
|---|---|
| fp32 | `Conv` 69, `MatMul` 48 |
| int8 | **`ConvInteger` 69**, `MatMulInteger` 42, `MatMul` 6, `DynamicQuantizeLinear` 80 |

**All 69 convolutions were quantized.** My pessimism came from a stale belief that
ORT dynamic quantization only handles MatMul/Attention — it handled every Conv in a
conv-heavy U-Net. The 80 `DynamicQuantizeLinear` nodes are genuine per-forward-pass
overhead, but nowhere near enough to cancel the gain.

### Result (base 25 / upsample 6, n=4 alternating, NEW seeds 303/404)

| | fp32 | int8 |
|---|---|---|
| upsample stage | 11.2s | **4.2s (2.7x faster)** |
| total, 2-image batch | 19.7s | **12.7s** |
| session load | 1.5s | 1.0s |
| artifact | 207MB | **53MB (3.9x smaller)** |
| **end-to-end via the API, 1 image** | 7.6s | **4.7s** |

**Invariant control says believe these numbers:** the base stage — identical in all 8
runs — measured 8.4-9.1s, a 1.08x spread. First genuinely quiet benchmark of the day.

**Quality: mean 24.0 dB** (24.2 / 25.1 / 22.3 / 24.5), mean |diff| ~12/255.

### The metric lesson, which matters more than the speedup

INT8 scores **24.0 dB — worse than the 15→6 step cut at 30.8 dB — yet looks
dramatically better.** Same metric, same scale, opposite direction of error:

> - **Step reduction removes detail → blur.** Perceptually obvious, cheap in MSE.
>   **PSNR understates it.**
> - **Quantization adds noise → fine grain on every pixel.** Expensive in MSE,
>   perceptually mild — on fur it even reads as texture. **PSNR overstates it.**
>
> A single scalar cannot rank two different *kinds* of degradation against each
> other. This is why the side-by-side grid is the real instrument and PSNR is only a
> drift detector. Anyone comparing "quality" with one number across dissimilar
> techniques is comparing nothing.

**Also invalidated:** the working assumption that "int8 will give pathetic images."
For *dynamic* int8 on this model it's false. Static quantization (with calibration)
would be the next thing to try and could be better still — or much worse, since it
fixes activation ranges from a calibration set.

**Deployment.** `UP_MODEL` is now a compose-level env var, default fp32. int8 is
opt-in:
```bash
UP_MODEL=/app/models/upsample_model.int8.onnx docker compose up -d
```
Precision is a deploy-time choice, not a code change — same treatment as step count.

**Cumulative: 39s → 4.7s (~8.3x), CPU-only, no GPU, no retraining.**
Step reduction (~5.1x) and int8 (~1.6x on top) compose because they attack different
things: fewer model calls, and cheaper calls.

---

## 2026-08-03 — Re-measured after optimizing, then added admission control

### Part 1: the load test had to be re-run, and the shape did not change

Every load number was measured at ~41s service time. After the step reduction it is
7.6s, so all of it was stale. **Optimizing without re-measuring the system-level
effect is where the discipline breaks** — the point of a baseline is to compare
against it. Same method as before (16 spoofed `X-Forwarded-For` identities, one burst):

| metric | before (50/50 steps) | after (25/6 steps) |
|---|---|---|
| service time | ~41s | **7.6s** |
| throughput ceiling | 1.57/min | **7.91/min** |
| avg queue wait | 157s | 56.1s |
| p50 / p95 total | — / — | 63.3s / **115.7s** |

> **Queue wait is still 7.4x service time** (56.1s vs 7.6s). A 5x faster model moved
> the ceiling and changed nothing about the *shape*: at saturation, latency is set by
> how many requests are ahead of you, not by how fast one is. **Optimization is not a
> substitute for load management** — that is why this needed a separate fix, and why
> "make the model faster" would never have closed it.

**Side effect: the `active:{ip}` slot-TTL bug stopped reproducing.** Max queue wait is
now 113.8s against a 300s TTL. **It is latent, not fixed** — it returns once queue
depth passes ~40. Worth recording, because "the bug went away" after an unrelated
change is exactly the kind of thing that gets mistaken for a fix.

### Part 2: admission control

The per-IP rate limit bounds **one** client. Nothing bounded **aggregate** load, so 16
compliant clients each got `200 OK` and then waited up to two minutes.

Added: a server-wide cap on `pending`+`started` jobs. Over the cap → `503` with a
`Retry-After` header and a body carrying `queue_depth` / `max_queue_depth` /
`retry_after_s`. Admitted requests now return `queue_position` and `est_wait_s` so
clients can back off instead of hot-polling. Two metrics: `diffusion_queue_depth`
(gauge) and `diffusion_admission_rejected_total` (counter). Read together they say
which knob to turn — rejections climbing with the queue *at* the cap means add
workers; rejections with a *shallow* queue means the cap is too low.

**Depth is derived state, not a counter.** A Redis `INCR`/`DECR` pair is cheaper but
leaks on every crash: one worker killed mid-job leaves the counter high forever and
the service refuses traffic permanently. Counting indexed rows is self-healing. The
count is windowed to `created_at >= now - 2*JOB_TIMEOUT_S` so rows stuck `started`
behind a dead worker cannot hold capacity forever (a reaper is still the proper fix;
this just makes the leak non-fatal).

**`SERVICE_TIME_S` is advisory only** — it feeds `Retry-After` and `est_wait_s`, never
the admission decision. If it drifts, estimates get wrong but the cap still works.
Control inputs and display values should not share a source of truth.

### The race, predicted and then measured

First implementation counted, *then* inserted in a separate transaction. Prediction
written down before testing: **"I expect the cap to leak under burst."** It did.

| config (identical 16-client burst) | admitted / shed | avg queue | max queue | p95 total |
|---|---|---|---|---|
| no admission control | 16 / 0 | 56.1s | 113.8s | 115.7s |
| **soft cap — raced** | **15 / 1** | 45.4s | 90.7s | 92.7s |
| **locked cap** | **12 / 4** | **35.9s** | **70.7s** | **73.5s** |

15 admitted against a cap of 12: all 16 requests read depth below the cap before any
row committed. Textbook time-of-check-to-time-of-use. **A cap that leaks 25% under
burst is not a cap**, and the latency benefit was correspondingly negligible.

Fix: `pg_advisory_xact_lock` around the decision, with the count and the `INSERT` in
**one** transaction. Both halves are load-bearing — the advisory lock releases at
commit, which is also when the row becomes visible to other transactions' counts, so
splitting them across two transactions would serialize nothing. A `503` rolls the
transaction back, so a rejected request leaves no row to inflate the next caller's
count (otherwise rejections cascade). Postgres-only; on the SQLite fallback the race
cannot occur — single writer.

**Result: exactly 12 admitted, 4 shed. p95 down 36%** (115.7 → 73.5s), and rejected
clients get an immediate answer with a retry hint instead of a two-minute wait.

### The pattern worth naming

> This is the **same class of bug** as the 2026-08-01 job-state guard: read-check-write
> that is only atomic if something serialises it. That one hid behind SQLite's file
> lock; this one hid behind low concurrency. Both were **found by load, not by reading
> code**, and both were fixed by making the read and the write one atomic step.
>
> Any time code reads state, decides, and then writes based on that decision, the
> question is not *"is the logic right"* but **"what stops two callers doing this at
> once?"** In this codebase the answer is now `SELECT ... FOR UPDATE` for a specific
> row and `pg_advisory_xact_lock` for a decision spanning many rows.

**Still open** (deliberately deferred): failure-injection drills + postmortems, CI,
alerting/SLOs, the slot-TTL fix, Alembic, retention, auth.

---

## 2026-08-01 — Merged README into this journal

`README.md` (models, theory, ONNX) was folded into Parts 1–6 here and deleted, on
request. Noting the trade-off for whoever reads this later: a repo with no README
gives a newcomer no entry point, and the reference material now lives under
`serving/` despite covering training and export too. Moving this file to
`Diffusion/JOURNAL.md` would fix the second half of that.

---

# Part 8 — Standing findings (carry forward)

- ~~fp16 models are exported and unused~~ → **MEASURED 2026-08-02: fp16 gives no
  latency win on CPU** (upsample 30.8s vs 32.1s median, inside noise at n=3). It
  halves the artifact (207→104MB) and slightly speeds session load. The artifact was
  never broken — `ORT_ENABLE_ALL` miscompiled it; `_make_session` now pins
  `ORT_ENABLE_EXTENDED`. Model paths are env-overridable (`BASE_MODEL` / `UP_MODEL`).
  **Re-run on GPU — this result should invert.** INT8 is the CPU-relevant one and is
  still unmeasured.
- ~~The rate limiter blocks load testing~~ → solved with per-client
  `X-Forwarded-For`, which works only because `--forwarded-allow-ips=*` trusts any
  client. **That is itself a hole to close** (narrow to the proxy IP).
- **No global admission control.** Per-IP limits don't bound aggregate load; 8
  compliant clients produced 382s tail latency with no rejection. Needs a
  queue-depth cap / backpressure.
- **`active:{ip}` slot TTL assumes zero queue wait** and expires mid-queue under
  load, breaking the latest-wins invariant.
- **GPU numbers are unobtainable here.** Docker Desktop on macOS gives containers no
  GPU passthrough, so this `onnxruntime` is permanently CPU-only. GPU measurement
  needs a cloud instance with `onnxruntime-gpu` + a CUDA base image — which the
  Dockerfile makes a config change rather than a rewrite.
- **Open bug, highest silent-damage potential:** no `UNIQUE (job_id, s3_key)` on
  `images`. With `task_acks_late=True`, a worker crash redelivers the task; the S3
  key is idempotent but the DB insert isn't, so a retry duplicates image rows.
- **Open bug:** `finished_at` is `NULL` on superseded jobs, so duration queries
  silently exclude them.
- **Next measurement, once the above is stable:** make model paths env-configurable
  and A/B fp32 vs fp16 on the **upsample** model only, under a fixed environment.

---

# Part 9 — How image quality was compared (methodology)

Kept as reference because every optimization above rests on it.

**The problem: there is no ground truth for a generated image.** No reference dog
exists, so *absolute* quality is not measurable this way. What is measurable is
**drift from a reference configuration.** Every number above means "this differs from
the reference by X", never "this is X good".

**What makes the pairs comparable:**
1. **Same seed** — identical initial noise.
2. **DDIM with `eta=0` is deterministic** — no sampling randomness, so a given seed +
   config always yields a bit-identical image. Re-runs prove nothing new.
3. **One variable at a time** — sweeping upsample steps pins base at 50; sweeping base
   pins upsample at 6; comparing precision pins both step counts.

Without those you are comparing two unrelated random images and any metric is noise.

**The metric — PSNR** on final uint8 RGB:
```
MSE  = mean over all pixels and channels of (a - b)**2
PSNR = 10 * log10(255**2 / MSE)      # higher = more similar; capped at 99 for identical
```
Log scale: **+6 dB ~ half the pixel error**. Conventional reading: >40 dB identical,
35-40 imperceptible, 30-35 slight, <30 visible.

**Reported alongside PSNR**, because one scalar hides failure modes: exact-equality
flag, `max|diff|`, `mean|diff|`, and **% of pixels differing**. That last one answered
"is the sampler even running?" — 15 vs 6 differed on **99.9% of pixels at mean
amplitude 7/255**: low amplitude everywhere = changed texture; a few pixels changing a
lot would mean localized artifacts.

**Anti-self-deception measures:**
- **Held-out seeds.** 6 steps was chosen looking at seeds 7/42, then re-validated on
  unseen 101/202 (30.8 dB vs 31.3 dB — held). int8 was tested only on fresh 303/404.
  Judging a choice on the images used to make it is circular.
- **Side-by-side grids at native 256x256**, same seed per column. The eye is the
  arbiter; PSNR only ranks.
- **An extreme control.** 2 steps → 16.8 dB and visible green mush. A silently broken
  step count would have left that control looking fine.
- **An invariant control.** A stage held constant across runs; its variance is a
  direct readout of environmental noise.

**Known weaknesses — state these before anyone else does:**
- PSNR is **pixel-wise and structure-blind**: shift an image one pixel and it collapses
  while looking identical.
- It punishes benign global shifts as hard as real artifacts.
- **It cannot compare across kinds of error** (the blur-vs-grain finding above).

Better tools not used, and why: **SSIM** (local structure — cheap, worth adding),
**LPIPS** (learned perceptual, closest to human judgment — needs a pretrained net),
**FID/KID** (distance between *distributions* of real vs generated — the metric papers
report, and the honest answer to "did quality drop", but needs thousands of samples
plus an Inception model). PSNR-against-a-reference answers the narrower question
*"how much did this config change the output"*, which is the right question for
choosing a setting — provided it isn't overclaimed as a quality score.
