"""Production-grade cascading diffusion inference pipeline (ONNX Runtime only).

Stage 1: base model      -> 64x64  class-conditioned image
Stage 2: upsampler model -> 256x256, conditioned on a bicubic upscale of stage 1

No TensorFlow at runtime — inference runs entirely on onnxruntime + numpy.
(TF is still used in inference.ipynb to *re-export* the .onnx files from Keras
weights; that is a build step, not a serving dependency.)

The two models do NOT share hyper-parameters:
  - base:      IMAGE_SIZE=64,  TIMESTEPS=1000, inputs [image, time, label]
  - upsampler: IMAGE_SIZE=256, TIMESTEPS=400,  inputs [image(6ch), time, blur, label]
Each therefore gets its own cosine noise schedule.

Run it: `python3 cascade_pipeline.py` — configuration lives in `main()`.
Use it as a library:

    pipe = CascadePipeline(CascadeConfig())
    result = pipe.generate(labels=[0, 1], seed=42)   # 0=dog, 1=cat
    result.save(Path("out"))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as rt
from PIL import Image

logger = logging.getLogger("cascade")

SCRIPT_DIR = Path(__file__).resolve().parent


class JobCancelled(Exception):
    """Raised from the sampling loop when should_cancel() fires (superseded job)."""


def _stage_timer(on_stage):
    """Context manager factory: times a block and reports it. No-op if on_stage is None.
    Deliberately reports on the exception path too — a cancelled stage's partial time
    is real work done, and hiding it would understate cost under load."""
    from contextlib import contextmanager, nullcontext
    from time import perf_counter

    if on_stage is None:
        return lambda _name: nullcontext()

    @contextmanager
    def timer(name):
        t0 = perf_counter()
        try:
            yield
        finally:
            on_stage(name, perf_counter() - t0)

    return timer


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CascadeConfig:
    """All knobs for the pipeline. Edit in main(); not a CLI."""

    channels: int = 3

    base_image_size: int = 64
    base_timesteps: int = 1000
    # Env-overridable so fp32/fp16/int8 variants can be A/B'd without a code edit
    # (e.g. UP_MODEL=/app/models/upsample_model.fp16.onnx). Resolved at import.
    base_model: Path = Path(os.getenv("BASE_MODEL", SCRIPT_DIR / "models" / "base_model.onnx"))

    up_image_size: int = 256
    up_timesteps: int = 400
    up_model: Path = Path(os.getenv("UP_MODEL", SCRIPT_DIR / "models" / "upsample_model.onnx"))

    # Sampler: "ddpm" (full, high quality, slow) or "ddim" (few-step, deterministic).
    # DDIM at ~50 steps is the production default: ~20x fewer model calls, near-identical quality.
    sampler: str = "ddim"
    # base model, from pure noise. 25 measured indistinguishable from 50 (2026-08-02,
    # 31.7 dB, same subjects); 15 starts corrupting eyes. Env: DDIM_STEPS.
    ddim_steps: int = int(os.getenv("DDIM_STEPS", "25"))
    # Upsampler steps are SEPARATE and deliberately fewer. It is a refiner conditioned
    # on a good 64x64 guide, not a generator from noise, so it converges in far fewer
    # steps — and at 83% of total cost it is the only stage where steps buy latency.
    # 6 chosen from the 2026-08-02 sweep: 8.3x fewer model calls in the stage that
    # dominates cost. 50/25/15 were mutually indistinguishable on AFHQ dog/cat.
    # Set None to fall back to ddim_steps (the old single-knob behaviour).
    up_ddim_steps: int | None = int(os.getenv("UP_DDIM_STEPS", "6"))
    ddim_eta: float = 0.0  # 0 = deterministic DDIM; ->1 recovers DDPM-like stochasticity

    @property
    def up_steps_effective(self) -> int:
        return self.ddim_steps if self.up_ddim_steps is None else self.up_ddim_steps

    # Preference order; unavailable ones are skipped, CPU is always appended.
    # CoreML is intentionally NOT default: its EP can't reliably run dynamic-batch
    # graphs (compiles but fails at runtime). Use CUDA on a server; CPU locally.
    # For CoreML specifically, export a static-shape fp16 MLProgram instead.
    providers: tuple[str, ...] = (
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


# --------------------------------------------------------------------------- #
# Cosine noise schedule (numpy port of the training schedule)
# --------------------------------------------------------------------------- #
def cosine_beta_schedule(timesteps: int) -> np.ndarray:
    s = 0.008
    t = np.arange(0, timesteps, dtype=np.float32)
    phase = ((t / timesteps) + s) / (1.0 + s)
    f_t = np.cos(phase * (np.pi / 2.0)) ** 2
    f_0 = np.cos((s / (1.0 + s)) * (np.pi / 2.0)) ** 2
    alpha_bar = f_t / f_0
    alpha_bar_shifted = np.concatenate([[1.0], alpha_bar[:-1]])
    beta = np.clip(1.0 - alpha_bar / alpha_bar_shifted, 0.0, 0.999)
    return beta.astype(np.float32)


class Schedule:
    """Cosine noise schedule + DDPM/DDIM reverse steps, all in numpy."""

    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        self.beta = cosine_beta_schedule(timesteps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = np.cumprod(self.alpha)

    def _abar(self, t: int) -> float:
        # t < 0 denotes the clean image (alpha_bar = 1), used as DDIM's final target.
        return float(self.alpha_bar[t]) if t >= 0 else 1.0

    def ddpm_step(self, x_t: np.ndarray, pred_noise: np.ndarray, t: int,
                  rng: np.random.Generator) -> np.ndarray:
        """One ancestral DDPM reverse step (stochastic, full T steps)."""
        alpha_t = self.alpha[t]
        alpha_t_bar = self.alpha_bar[t]
        eps_coef = (1.0 - alpha_t) / np.sqrt(1.0 - alpha_t_bar)
        mean = (1.0 / np.sqrt(alpha_t)) * (x_t - eps_coef * pred_noise)
        var = self.beta[t]
        z = rng.standard_normal(x_t.shape).astype(np.float32)
        noise_mask = 1.0 if t > 0 else 0.0  # no noise added on the final step
        return (mean + noise_mask * np.sqrt(var) * z).astype(np.float32)

    def ddim_step(self, x_t: np.ndarray, pred_noise: np.ndarray, t: int, t_prev: int,
                  rng: np.random.Generator, eta: float = 0.0, clip: bool = True) -> np.ndarray:
        """One DDIM reverse step from t -> t_prev (Song et al. 2021).

        eta=0 is deterministic (implicit); eta=1 reproduces DDPM's noise level.
        Jumping t->t_prev over a subsequence is what makes DDIM few-step.
        """
        ab_t = self._abar(t)
        ab_prev = self._abar(t_prev)
        # Predict x_0 from x_t and the noise estimate, then re-noise toward t_prev.
        x0 = (x_t - np.sqrt(1.0 - ab_t) * pred_noise) / np.sqrt(ab_t)
        if clip:
            x0 = np.clip(x0, -1.0, 1.0)  # models trained on [-1,1]; keeps early steps stable
        sigma = 0.0
        if eta > 0.0:
            sigma = eta * np.sqrt((1.0 - ab_prev) / (1.0 - ab_t)) * np.sqrt(1.0 - ab_t / ab_prev)
        dir_coef = np.sqrt(max(1.0 - ab_prev - sigma ** 2, 0.0))
        x_prev = np.sqrt(ab_prev) * x0 + dir_coef * pred_noise
        if sigma > 0.0 and t_prev > 0:
            x_prev = x_prev + sigma * rng.standard_normal(x_t.shape).astype(np.float32)
        return x_prev.astype(np.float32)


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def bicubic_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bicubic resize of a single HWC float image, per-channel in Pillow "F"
    mode so the [-1, 1] range is preserved (no uint8 quantisation)."""
    out = np.empty((size, size, img.shape[-1]), dtype=np.float32)
    for c in range(img.shape[-1]):
        pil = Image.fromarray(img[..., c], mode="F").resize((size, size), Image.BICUBIC)
        out[..., c] = np.asarray(pil, dtype=np.float32)
    return out


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Single HWC image in [-1, 1] -> uint8 [0, 255]."""
    return (np.clip((img + 1.0) / 2.0, 0.0, 1.0) * 255.0).round().astype(np.uint8)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class CascadeResult:
    labels: list[int]
    base: np.ndarray        # (N, 64, 64, 3)   in [-1, 1]
    condition: np.ndarray   # (N, 256, 256, 3) in [-1, 1] (bicubic upscale)
    final: np.ndarray       # (N, 256, 256, 3) in [-1, 1]

    def save(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i, label in enumerate(self.labels):
            path = out_dir / f"cascade_label{label}_{i}.png"
            Image.fromarray(to_uint8(self.final[i])).save(path)
            paths.append(path)
        logger.info("saved %d image(s) to %s", len(paths), out_dir)
        return paths


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class CascadePipeline:
    def __init__(self, config: CascadeConfig = CascadeConfig()):
        self.cfg = config
        self.base_sess = self._make_session(config.base_model)
        self.up_sess = self._make_session(config.up_model)
        self.base_sched = Schedule(config.base_timesteps)
        self.up_sched = Schedule(config.up_timesteps)
        self._base_inputs = {i.name for i in self.base_sess.get_inputs()}
        self._up_inputs = {i.name for i in self.up_sess.get_inputs()}
        _require({"image_input", "time_input", "label_input"}, self._base_inputs, "base")
        _require({"image_input", "time_input", "blur_input", "label_input"},
                 self._up_inputs, "upsampler")

    def _make_session(self, model_path: Path) -> rt.InferenceSession:
        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found: {model_path}. Run export_onnx.py first."
            )
        available = set(rt.get_available_providers())
        providers = [p for p in self.cfg.providers if p in available]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")

        # Pin the optimization level. The Python SessionOptions default is
        # ORT_ENABLE_ALL, whose layout optimizations MISCOMPILE the fp16 upsampler on
        # onnxruntime 1.28/aarch64: it produces a Gather over an empty tensor
        # ("indices element out of data bounds ... range [0,-1]") and fails on the
        # first forward pass. EXTENDED keeps every fusion that matters and is stable.
        # Never rely on the default here — it changes between ORT releases.
        so = rt.SessionOptions()
        so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        try:
            sess = rt.InferenceSession(str(model_path), so, providers=providers)
        except Exception as exc:
            # Some EPs (e.g. CoreML) fail to compile certain dynamic-shape graphs.
            # Degrade to CPU rather than crash the whole pipeline.
            logger.warning("%s failed on %s (%s); falling back to CPU",
                           model_path.name, providers[0], type(exc).__name__)
            sess = rt.InferenceSession(str(model_path), so,
                                       providers=["CPUExecutionProvider"])
        logger.info("loaded %s on %s", model_path.name, sess.get_providers()[0])
        return sess

    def _predict_noise(self, sess, x, t, labels, cond, blur):
        """One model forward pass at timestep t for the whole batch."""
        t_arr = np.full((labels.shape[0], 1), float(t), dtype=np.float32)
        model_in = x if cond is None else np.concatenate([x, cond], axis=-1)
        feed = {"image_input": model_in, "time_input": t_arr, "label_input": labels}
        if cond is not None:
            feed["blur_input"] = blur
        return sess.run(None, feed)[0]

    def _sample(self, sess, sched, x, labels, cond, rng, desc, should_cancel=None,
                steps=None):
        """Reverse-diffuse `x` to a clean image using the configured sampler.

        Both samplers stop before a true t=0 step: the cosine schedule has
        alpha_bar[0]==1, so a t=0 DDPM step divides by sqrt(1-alpha_bar_0)=0.
        DDIM instead targets t_prev=-1 (the clean image, alpha_bar=1) on its last step.

        `should_cancel`, if given, is polled between steps; when it returns True the
        loop raises JobCancelled so the caller can stop a superseded job cheaply
        (bounded to one denoising step of latency — no process kill).
        """
        cfg = self.cfg
        blur = np.zeros((labels.shape[0], 1), dtype=np.float32)  # clean guide at inference
        steps = cfg.ddim_steps if steps is None else steps

        if cfg.sampler == "ddim":
            # Descending subsequence of timesteps: T-1 -> 0, `ddim_steps` hops.
            times = np.linspace(sched.timesteps - 1, 0, steps + 1).round().astype(int)
            for i in range(steps):
                if should_cancel and should_cancel():
                    raise JobCancelled(f"{desc} cancelled at ddim step {i}")
                t, t_prev = int(times[i]), int(times[i + 1])
                if t == t_prev:
                    continue  # can happen if steps ~ timesteps; skip no-op hop
                eps = self._predict_noise(sess, x, t, labels, cond, blur)
                # Last hop targets the clean image (t_prev=-1) instead of t=0.
                x = sched.ddim_step(x, eps, t, t_prev if i < steps - 1 else -1,
                                    rng, cfg.ddim_eta)
                logger.debug("%s ddim %d/%d", desc, i + 1, steps)
        elif cfg.sampler == "ddpm":
            for i in range(sched.timesteps - 1):
                # Poll every 20 steps: 1000-step DDPM shouldn't hammer Redis.
                if should_cancel and i % 20 == 0 and should_cancel():
                    raise JobCancelled(f"{desc} cancelled at ddpm step {i}")
                t = sched.timesteps - i - 1
                eps = self._predict_noise(sess, x, t, labels, cond, blur)
                x = sched.ddpm_step(x, eps, t, rng)
                if i % 100 == 0:
                    logger.debug("%s ddpm %d/%d", desc, i, sched.timesteps - 1)
        else:
            raise ValueError(f"unknown sampler {cfg.sampler!r}; use 'ddpm' or 'ddim'")
        return x

    def generate(self, labels: list[int], seed: int | None = None,
                 should_cancel=None, on_stage=None) -> CascadeResult:
        """Run the full cascade for a batch of class ids in a single pass per stage.

        `should_cancel`: optional no-arg callable polled between denoising steps;
        raise-based cooperative cancellation (see `_sample`).
        `on_stage`: optional (stage_name, seconds) callback per cascade stage, so a
        caller can record timings without this module knowing about metrics.
        """
        if not labels:
            raise ValueError("labels must be a non-empty list of class ids")
        cfg = self.cfg
        timer = _stage_timer(on_stage)
        rng = np.random.default_rng(seed)
        lab = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
        n = lab.shape[0]

        logger.info("[stage 1] base %dx%d, %d image(s), sampler=%s", cfg.base_image_size,
                    cfg.base_image_size, n, cfg.sampler)
        x = rng.standard_normal(
            (n, cfg.base_image_size, cfg.base_image_size, cfg.channels)
        ).astype(np.float32)
        with timer("base"):
            base = self._sample(self.base_sess, self.base_sched, x, lab, None, rng, "base",
                                should_cancel, steps=cfg.ddim_steps)

        with timer("condition"):
            cond = np.stack([bicubic_resize(base[i], cfg.up_image_size) for i in range(n)])

        logger.info("[stage 2] upsample -> %dx%d, sampler=%s", cfg.up_image_size,
                    cfg.up_image_size, cfg.sampler)
        x = rng.standard_normal(
            (n, cfg.up_image_size, cfg.up_image_size, cfg.channels)
        ).astype(np.float32)
        with timer("upsample"):
            final = self._sample(self.up_sess, self.up_sched, x, lab, cond, rng, "upsample",
                                 should_cancel, steps=cfg.up_steps_effective)

        return CascadeResult(labels=list(labels), base=base, condition=cond, final=final)


def _require(needed: set[str], present: set[str], name: str) -> None:
    missing = needed - present
    if missing:
        raise ValueError(f"{name} model is missing inputs {missing}; got {present}")


# --------------------------------------------------------------------------- #
# Entry point — configure here, no CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = CascadeConfig()
    labels = [0, 1]          # 0 = dog, 1 = cat
    seed = 42
    out_dir = SCRIPT_DIR / "cascade_output"

    pipe = CascadePipeline(config)
    result = pipe.generate(labels=labels, seed=seed)
    result.save(out_dir)


if __name__ == "__main__":
    main()
