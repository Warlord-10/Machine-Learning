"""Self-checks for the parts of cascade_pipeline that don't need the 200MB
ONNX models: the noise schedule, the reverse step, and the resize helper.

Run: python3 test_cascade_pipeline.py
"""

import numpy as np

from cascade_pipeline import NoiseSchedule, bicubic_resize, to_uint8


def test_schedule():
    s = NoiseSchedule(1000)
    assert s.betas.shape == (1000,)
    assert np.all(s.betas >= 0.0) and np.all(s.betas <= 0.999)
    # alpha_bar is a cumulative product of alpha<1 -> monotonically non-increasing.
    assert np.all(np.diff(s.alpha_bars) <= 1e-6)
    # Cosine schedule pins alpha_bar[0] to 1.0 (the div-by-zero the loop avoids).
    assert abs(s.alpha_bars[0] - 1.0) < 1e-5


def test_ddpm_step_no_noise_at_zero():
    s = NoiseSchedule(400)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    noise = np.zeros_like(x)
    # At t=0 the noise term is masked out, so the step is deterministic given x.
    a = s.ddpm_step(x, noise, 0, np.random.default_rng(1))
    b = s.ddpm_step(x, noise, 0, np.random.default_rng(2))
    assert np.allclose(a, b), "t=0 step must not inject noise"
    # At t>0 the two RNGs must diverge.
    c = s.ddpm_step(x, noise, 5, np.random.default_rng(1))
    d = s.ddpm_step(x, noise, 5, np.random.default_rng(2))
    assert not np.allclose(c, d), "t>0 step must be stochastic"


def test_bicubic_resize_preserves_range():
    img = np.linspace(-1.0, 1.0, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
    out = bicubic_resize(img, 256)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.float32
    # Bicubic can overshoot slightly, but stays near the source range (not uint8-clipped).
    assert out.min() < -0.5 and out.max() > 0.5


def test_to_uint8():
    img = np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32)
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert list(out[0, 0]) == [0, 128, 255]


def test_ddim_final_step_returns_x0():
    # On the last hop (t_prev=-1, alpha_bar=1) with eta=0, DDIM must return the
    # clipped x0 estimate exactly — that's the clean image the loop converges to.
    s = NoiseSchedule(400)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    eps = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    t = 100
    ab_t = s.alpha_bars[t]
    x0 = np.clip((x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t), -1.0, 1.0)
    out = s.ddim_step(x, eps, t, -1, rng, eta=0.0)
    assert np.allclose(out, x0, atol=1e-5), "final DDIM step must yield clipped x0"


def test_ddim_deterministic_when_eta_zero():
    s = NoiseSchedule(400)
    x = np.random.default_rng(0).standard_normal((1, 8, 8, 3)).astype(np.float32)
    eps = np.random.default_rng(1).standard_normal((1, 8, 8, 3)).astype(np.float32)
    a = s.ddim_step(x, eps, 200, 150, np.random.default_rng(7), eta=0.0)
    b = s.ddim_step(x, eps, 200, 150, np.random.default_rng(99), eta=0.0)
    assert np.allclose(a, b), "eta=0 DDIM must be deterministic (rng-independent)"


if __name__ == "__main__":
    test_schedule()
    test_ddpm_step_no_noise_at_zero()
    test_bicubic_resize_preserves_range()
    test_to_uint8()
    test_ddim_final_step_returns_x0()
    test_ddim_deterministic_when_eta_zero()
    print("all self-checks passed")
