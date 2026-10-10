"""Export the trained Keras diffusion models to ONNX with a DYNAMIC batch dim.

This is a one-off build step (run once per trained checkpoint), not a serving
dependency. Fixes the old batch-locked-to-1 export by declaring the batch
dimension as `None`, and optionally emits an FP16 copy for faster inference.

Output:
  models/base_model.onnx,      models/upsample_model.onnx        (FP32)
  models/base_model.fp16.onnx, models/upsample_model.fp16.onnx   (if EXPORT_FP16)

Run: python3 export_onnx.py     (configuration is in main(), no CLI)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as rt
import tensorflow as tf
import tf2onnx
from onnxruntime.transformers.float16 import convert_float_to_float16

import _base_arch
import _up_arch

logger = logging.getLogger("export")

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"
MODELS_DIR = SCRIPT_DIR / "models"

OPSET = 18
PARITY_TOL = 1e-3       # FP32 ONNX must match Keras this closely, or the export is untrustworthy.
EXPORT_FP16 = True      # also write a *.fp16.onnx copy (weights/compute in half precision)
FP16_TOL = 0.05         # FP16 is lossy; allow a small, harmless divergence (DDPM/DDIM self-correct).


def build_base() -> tf.keras.Model:
    tf.keras.backend.clear_session()  # reset layer-name counter to match saved weights
    model = _base_arch.build_unet(64)
    model.load_weights(str(WEIGHTS_DIR / "multiclass-new.weights.h5"))
    return model


def build_upsampler() -> tf.keras.Model:
    tf.keras.backend.clear_session()
    model = _up_arch.build_unet(256)
    model.load_weights(str(WEIGHTS_DIR / "upsampler_checkpoint.weights.h5"))
    return model


def export(model: tf.keras.Model, out_path: Path, opset: int = OPSET) -> None:
    """Export with a dynamic batch dim: the leading axis is `None`, so the ONNX
    graph accepts any batch size at inference."""
    specs = [
        tf.TensorSpec([None] + list(inp.shape[1:]), inp.dtype, name=inp.name.split(":")[0])
        for inp in model.inputs
    ]

    @tf.function(input_signature=[specs])
    def run(inputs):
        return model(inputs, training=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_function(
        run, input_signature=[specs], opset=opset, output_path=str(out_path)
    )
    logger.info("exported %s", out_path)


def parity_check(model: tf.keras.Model, onnx_path: Path, sample_inputs: list[np.ndarray]) -> None:
    """Run ONNX and Keras on the SAME (batch=2) input and assert they agree.
    The batch=2 input also proves the dynamic batch dim actually works."""
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_shapes = [i.shape for i in sess.get_inputs()]
    feed = {i.name: v for i, v in zip(sess.get_inputs(), sample_inputs)}
    onnx_out = sess.run(None, feed)[0]
    keras_out = model(sample_inputs, training=False).numpy()
    diff = float(np.abs(onnx_out - keras_out).max())

    logger.info("  onnx input shapes: %s", in_shapes)
    logger.info("  batch=2 out shape: %s | max|onnx-keras| = %.2e", onnx_out.shape, diff)
    assert onnx_out.shape == keras_out.shape, "shape mismatch"
    assert diff < PARITY_TOL, f"ONNX diverges from Keras ({diff:.2e} >= {PARITY_TOL})"
    # Guard the actual point of this re-export: batch must be dynamic, not 1.
    assert in_shapes[0][0] in (None, "unk__0", -1) or not isinstance(in_shapes[0][0], int), (
        f"batch dim is static ({in_shapes[0][0]}); export is still batch-locked"
    )
    logger.info("  parity OK, batch dim is dynamic")


def convert_fp16(fp32_path: Path, fp16_path: Path) -> None:
    """Write an FP16 copy of an ONNX model. `keep_io_types=True` leaves the graph
    inputs/outputs as float32, so callers (cascade_pipeline.py) need no changes —
    only the interior weights/compute go to half precision."""
    model_fp16 = convert_float_to_float16(onnx.load(str(fp32_path)), keep_io_types=True)
    onnx.save(model_fp16, str(fp16_path))
    logger.info("exported %s", fp16_path)


def fp16_parity(fp32_path: Path, fp16_path: Path, sample_inputs: list[np.ndarray]) -> None:
    """Compare FP16 against its FP32 source (not Keras) at a looser tolerance."""
    feed_names = [i.name for i in rt.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]).get_inputs()]
    feed = {n: v for n, v in zip(feed_names, sample_inputs)}
    o32 = rt.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    o16 = rt.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"]).run(None, feed)[0]
    diff = float(np.abs(o32 - o16).max())
    shrink = fp32_path.stat().st_size / fp16_path.stat().st_size
    logger.info("  fp16 max|Δ| vs fp32 = %.2e | %.2fx smaller", diff, shrink)
    assert diff < FP16_TOL, f"FP16 diverges too much ({diff:.2e} >= {FP16_TOL})"
    logger.info("  fp16 parity OK")


def process(name: str, model: tf.keras.Model, onnx_path: Path,
            sample_inputs: list[np.ndarray]) -> None:
    """Full export flow for one model: FP32 export + parity, then optional FP16."""
    logger.info("[%s] exporting FP32", name)
    export(model, onnx_path)
    parity_check(model, onnx_path, sample_inputs)
    if EXPORT_FP16:
        logger.info("[%s] converting FP16", name)
        fp16_path = onnx_path.with_suffix(".fp16.onnx")
        convert_fp16(onnx_path, fp16_path)
        fp16_parity(onnx_path, fp16_path, sample_inputs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(0)

    process("base", build_base(), MODELS_DIR / "base_model.onnx", [
        rng.standard_normal((2, 64, 64, 3)).astype(np.float32),   # image
        np.array([[10.0], [20.0]], dtype=np.float32),             # time
        np.array([[0.0], [1.0]], dtype=np.float32),               # label
    ])

    process("upsampler", build_upsampler(), MODELS_DIR / "upsample_model.onnx", [
        rng.standard_normal((2, 256, 256, 6)).astype(np.float32),  # image (6ch)
        np.array([[10.0], [20.0]], dtype=np.float32),              # time
        np.array([[0.0], [0.0]], dtype=np.float32),                # blur
        np.array([[0.0], [1.0]], dtype=np.float32),                # label
    ])

    logger.info("done -> %s", MODELS_DIR)


if __name__ == "__main__":
    main()
