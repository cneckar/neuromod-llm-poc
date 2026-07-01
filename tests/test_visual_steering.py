"""
Unit tests for the diffusion latent-steering math (issue: real latent steering).

Torch-free: only the pure scaling/direction math is tested here (the UNet hook attachment
requires a live model + GPU). Loaded by file path to avoid the torch-importing neuromod
package __init__.
"""

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "visual_steering", os.path.join(_HERE, "..", "neuromod", "visual_steering.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vs = _load()


def test_unit_direction_is_unit_and_deterministic():
    d = vs.unit_direction(16, seed=7)
    assert d.shape == (16,)
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(d, vs.unit_direction(16, seed=7))  # deterministic
    assert not np.allclose(d, vs.unit_direction(16, seed=8))  # seed-sensitive


def test_apply_steer_adds_scaled_direction():
    x = np.zeros((1, 4, 2, 2))
    direction = np.ones((1, 4, 1, 1))
    out = vs.apply_steer(x, direction, steer_scale=0.5, noise=None, noise_scale=0.0)
    assert np.allclose(out, 0.5)


def test_apply_steer_scales_monotonically():
    rng = np.random.RandomState(0)
    x = rng.randn(1, 8, 3, 3)
    direction = vs.unit_direction(8, 1).reshape(1, 8, 1, 1)
    lo = vs.apply_steer(x, direction, 0.1, None, 0.0)
    hi = vs.apply_steer(x, direction, 0.9, None, 0.0)
    # Larger steer_scale moves further from the original activation.
    assert np.linalg.norm(hi - x) > np.linalg.norm(lo - x)


def test_apply_steer_noise_term():
    x = np.zeros((1, 4, 2, 2))
    direction = np.zeros((1, 4, 1, 1))
    noise = np.ones((1, 4, 2, 2))
    out = vs.apply_steer(x, direction, 0.0, noise, noise_scale=0.3)
    assert np.allclose(out, 0.3)


def test_apply_steer_zero_scales_is_identity():
    x = np.random.RandomState(1).randn(1, 4, 2, 2)
    out = vs.apply_steer(x, np.ones((1, 4, 1, 1)), 0.0, np.ones_like(x), 0.0)
    assert np.allclose(out, x)


def test_stable_seed_deterministic_and_distinct():
    assert vs.stable_seed("dmt") == vs.stable_seed("dmt")
    assert vs.stable_seed("dmt") != vs.stable_seed("lsd")
    assert isinstance(vs.stable_seed("cocaine"), int)


def test_steering_context_inactive_when_zero():
    # active flag is False when both scales are zero (so we skip hooking entirely).
    ctx = vs.UNetActivationSteering(unet=None, steer_scale=0.0, noise_scale=0.0)
    assert ctx.active is False
    ctx2 = vs.UNetActivationSteering(unet=None, steer_scale=0.2, noise_scale=0.0)
    assert ctx2.active is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
