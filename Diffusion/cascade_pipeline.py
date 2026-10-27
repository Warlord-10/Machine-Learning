"""Cascading diffusion inference on ONNX Runtime (no TensorFlow at runtime).

Two class-conditioned models run in sequence:
  1. Base model      noise -> 64x64 image
  2. Upsample model  noise -> 256x256 image, guided by a bicubic upscale of stage 1

Each model was trained with its own cosine noise schedule (1000 and 400 timesteps),
so each carries its own `NoiseSchedule`.

Usage:
    pipeline = CascadePipeline(CascadeConfig())
    result = pipeline.generate(labels=[DOG, CAT], seed=42)
    result.save(Path("out"))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger("cascade")

MODELS_DIR = Path(__file__).resolve().parent / "models"
CHANNELS = 3
DOG, CAT = 0, 1
SAMPLERS = ("ddim", "ddpm")


@dataclass(frozen=True)
class CascadeConfig:
    # Env-overridable so fp32/fp16/int8 exports can be swapped without a code change.
    base_model_path: Path = Path(os.getenv("BASE_MODEL", MODELS_DIR / "base_model.onnx"))
    upsample_model_path: Path = Path(os.getenv("UP_MODEL", MODELS_DIR / "upsample_model.onnx"))

    base_image_size: int = 64
    upsample_image_size: int = 256
    # Must match training; these define each model's noise schedule.
    base_timesteps: int = 1000
    upsample_timesteps: int = 400

    # "ddim": few deterministic steps (production). "ddpm": every timestep, slow.
    sampler: str = "ddim"
    # DDIM steps per stage. Base: 25 is indistinguishable from 50, 15 corrupts eyes.
    # Upsample: it refines a good 64x64 guide and is ~83% of the cost; 6 is
    # indistinguishable from 50. Measured 2026-08-02, see serving/JOURNAL.md.
    base_steps: int = int(os.getenv("DDIM_STEPS", "25"))
    upsample_steps: int = int(os.getenv("UP_DDIM_STEPS", "6"))
    ddim_eta: float = 0.0  # 0 = deterministic; 1 = DDPM-level noise

    # Preference order; unavailable providers are skipped and CPU is always the fallback.
    # CoreML is left out on purpose: it compiles our dynamic-batch graphs but fails at runtime.
    providers: tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")

    def __post_init__(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {self.sampler!r}")


@dataclass
class CascadeResult:
    labels: list[int]
    base_images: np.ndarray  # (N, 64, 64, 3) in [-1, 1]
    images: np.ndarray       # (N, 256, 256, 3) in [-1, 1], the final output

    def save(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, (label, image) in enumerate(zip(self.labels, self.images)):
            path = out_dir / f"cascade_label{label}_{index}.png"
            Image.fromarray(to_uint8(image)).save(path)
            paths.append(path)
        return paths


class NoiseSchedule:
    """Cosine noise schedule (numpy port of the training one) and the reverse steps."""

    def __init__(self, timesteps: int, offset: float = 0.008):
        self.timesteps = timesteps
        steps = np.arange(0, timesteps, dtype=np.float32)
        phase = ((steps / timesteps) + offset) / (1.0 + offset)
        signal = np.cos(phase * (np.pi / 2.0)) ** 2
        signal_at_zero = np.cos((offset / (1.0 + offset)) * (np.pi / 2.0)) ** 2
        alpha_bars = signal / signal_at_zero
        previous_alpha_bars = np.concatenate([[1.0], alpha_bars[:-1]])

        self.betas = np.clip(1.0 - alpha_bars / previous_alpha_bars, 0.0, 0.999).astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def alpha_bar(self, timestep: int) -> float:
        # timestep -1 stands for the clean image (alpha_bar = 1): DDIM's final target.
        return float(self.alpha_bars[timestep]) if timestep >= 0 else 1.0

    def ddpm_step(self, noisy: np.ndarray, predicted_noise: np.ndarray, timestep: int,
                  rng: np.random.Generator) -> np.ndarray:
        """One stochastic DDPM step: timestep -> timestep - 1."""
        alpha = self.alphas[timestep]
        noise_scale = (1.0 - alpha) / np.sqrt(1.0 - self.alpha_bars[timestep])
        mean = (1.0 / np.sqrt(alpha)) * (noisy - noise_scale * predicted_noise)
        fresh_noise = rng.standard_normal(noisy.shape).astype(np.float32)
        add_noise = 1.0 if timestep > 0 else 0.0  # the last step adds none
        return (mean + add_noise * np.sqrt(self.betas[timestep]) * fresh_noise).astype(np.float32)

    def ddim_step(self, noisy: np.ndarray, predicted_noise: np.ndarray, timestep: int,
                  next_timestep: int, rng: np.random.Generator, eta: float = 0.0) -> np.ndarray:
        """One DDIM step (Song et al. 2021). It can jump many timesteps at once,
        which is what makes DDIM few-step."""
        alpha_bar = self.alpha_bar(timestep)
        next_alpha_bar = self.alpha_bar(next_timestep)

        # Estimate the clean image, then re-noise it to the next timestep's level.
        clean = (noisy - np.sqrt(1.0 - alpha_bar) * predicted_noise) / np.sqrt(alpha_bar)
        clean = np.clip(clean, -1.0, 1.0)  # trained on [-1, 1]; keeps early steps stable

        sigma = 0.0
        if eta > 0.0:
            sigma = (eta * np.sqrt((1.0 - next_alpha_bar) / (1.0 - alpha_bar))
                     * np.sqrt(1.0 - alpha_bar / next_alpha_bar))
        noise_direction = np.sqrt(max(1.0 - next_alpha_bar - sigma ** 2, 0.0))
        result = np.sqrt(next_alpha_bar) * clean + noise_direction * predicted_noise
        if sigma > 0.0 and next_timestep > 0:
            result = result + sigma * rng.standard_normal(noisy.shape).astype(np.float32)
        return result.astype(np.float32)


class DiffusionModel:
    """An ONNX noise-prediction model together with the schedule it was trained on."""

    def __init__(self, path: Path, timesteps: int, providers: tuple[str, ...],
                 is_upsampler: bool):
        self.session = _load_session(path, providers)
        self.schedule = NoiseSchedule(timesteps)
        self.is_upsampler = is_upsampler

        expected = {"image_input", "time_input", "label_input"}
        if is_upsampler:
            expected.add("blur_input")
        actual = {model_input.name for model_input in self.session.get_inputs()}
        if missing := expected - actual:
            raise ValueError(f"{path.name} is missing inputs {missing}; has {actual}")

    def predict_noise(self, images: np.ndarray, timestep: int, labels: np.ndarray,
                      guide: np.ndarray | None = None) -> np.ndarray:
        batch_size = labels.shape[0]
        inputs = {
            "image_input": images,
            "time_input": np.full((batch_size, 1), float(timestep), dtype=np.float32),
            "label_input": labels,
        }
        if self.is_upsampler:
            # The upsampler sees noise and guide stacked as 6 channels. blur_input is
            # the guide's blur level from training augmentation: 0 = clean guide.
            inputs["image_input"] = np.concatenate([images, guide], axis=-1)
            inputs["blur_input"] = np.zeros((batch_size, 1), dtype=np.float32)
        return self.session.run(None, inputs)[0]


class CascadePipeline:
    def __init__(self, config: CascadeConfig = CascadeConfig()):
        self.config = config
        self.base_model = DiffusionModel(config.base_model_path, config.base_timesteps,
                                         config.providers, is_upsampler=False)
        self.upsample_model = DiffusionModel(config.upsample_model_path, config.upsample_timesteps,
                                             config.providers, is_upsampler=True)

    def generate(self, labels: list[int], seed: int | None = None) -> CascadeResult:
        """Generate one 256x256 image per class label. Same seed -> same images."""
        if not labels:
            raise ValueError("labels must be a non-empty list of class ids")
        config = self.config
        rng = np.random.default_rng(seed)
        label_batch = np.asarray(labels, dtype=np.float32).reshape(-1, 1)

        logger.info("[stage 1] base %dpx, %d image(s), sampler=%s",
                    config.base_image_size, len(labels), config.sampler)
        noise = _gaussian_noise(rng, len(labels), config.base_image_size)
        base_images = self._denoise(self.base_model, noise, label_batch,
                                    config.base_steps, rng)

        logger.info("[stage 2] upsample -> %dpx", config.upsample_image_size)
        guide = np.stack([bicubic_resize(image, config.upsample_image_size)
                          for image in base_images])
        noise = _gaussian_noise(rng, len(labels), config.upsample_image_size)
        images = self._denoise(self.upsample_model, noise, label_batch,
                               config.upsample_steps, rng, guide)

        return CascadeResult(labels=list(labels), base_images=base_images, images=images)

    def _denoise(self, model: DiffusionModel, images: np.ndarray, labels: np.ndarray,
                 ddim_steps: int, rng: np.random.Generator,
                 guide: np.ndarray | None = None) -> np.ndarray:
        """Run the reverse diffusion from pure noise to a clean image.

        Neither sampler takes a real t=0 step: the cosine schedule has alpha_bar[0] = 1,
        which would divide by zero. DDIM's last step targets -1 (the clean image) instead.
        """
        schedule = model.schedule

        if self.config.sampler == "ddpm":
            for timestep in range(schedule.timesteps - 1, 0, -1):
                predicted_noise = model.predict_noise(images, timestep, labels, guide)
                images = schedule.ddpm_step(images, predicted_noise, timestep, rng)
            return images

        # DDIM: `ddim_steps` evenly spaced jumps from the last timestep down to 0.
        timesteps = np.linspace(schedule.timesteps - 1, 0, ddim_steps + 1).round().astype(int)
        for step in range(ddim_steps):
            timestep, next_timestep = int(timesteps[step]), int(timesteps[step + 1])
            if timestep == next_timestep:
                continue  # rounding duplicate when ddim_steps is close to timesteps
            if step == ddim_steps - 1:
                next_timestep = -1  # land on the clean image
            predicted_noise = model.predict_noise(images, timestep, labels, guide)
            images = schedule.ddim_step(images, predicted_noise, timestep, next_timestep,
                                        rng, self.config.ddim_eta)
        return images


def bicubic_resize(image: np.ndarray, size: int) -> np.ndarray:
    """Resize one HWC float image, channel by channel in Pillow's float mode so
    values stay in [-1, 1] instead of being quantised to uint8."""
    resized = np.empty((size, size, image.shape[-1]), dtype=np.float32)
    for channel in range(image.shape[-1]):
        plane = Image.fromarray(image[..., channel], mode="F")
        resized[..., channel] = np.asarray(plane.resize((size, size), Image.BICUBIC),
                                           dtype=np.float32)
    return resized


def to_uint8(image: np.ndarray) -> np.ndarray:
    """One HWC image in [-1, 1] -> uint8 pixels in [0, 255]."""
    return (np.clip((image + 1.0) / 2.0, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def _gaussian_noise(rng: np.random.Generator, batch_size: int, image_size: int) -> np.ndarray:
    return rng.standard_normal((batch_size, image_size, image_size, CHANNELS)).astype(np.float32)


def _load_session(path: Path, preferred_providers: tuple[str, ...]) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(f"ONNX model not found: {path}. Run scripts/export_onnx.py first.")

    available = set(ort.get_available_providers())
    providers = [provider for provider in preferred_providers if provider in available]
    if "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")

    # Pin the optimisation level; never rely on the default. ORT_ENABLE_ALL (the default)
    # miscompiles the fp16 upsampler on onnxruntime 1.28/aarch64 ("indices element out
    # of data bounds" on the first run). EXTENDED keeps the fusions that matter.
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    try:
        session = ort.InferenceSession(str(path), options, providers=providers)
    except Exception as error:
        # A GPU provider can fail to compile a graph; degrade to CPU instead of crashing.
        logger.warning("%s failed on %s (%s); falling back to CPU",
                       path.name, providers[0], type(error).__name__)
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    logger.info("loaded %s on %s", path.name, session.get_providers()[0])
    return session


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = CascadePipeline().generate(labels=[DOG, CAT], seed=42)
    paths = result.save(Path(__file__).resolve().parent / "cascade_output")
    logger.info("saved %s", ", ".join(map(str, paths)))
