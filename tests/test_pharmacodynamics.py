"""
Synthetic-tensor unit tests for the pharmacodynamic metrics layer.

These tests are GPU-free and torch-free: they validate the always-available metrics
(FFT scalars, spatial variance, SSIM, spectral entropy, inter-seed diversity fallback,
and the per-generation bundle) on crafted inputs with known answers.

The metrics module is loaded directly by file path so the tests do NOT trigger the heavy
``neuromod`` package ``__init__`` (which imports torch). In a full GPU environment the
normal ``from neuromod.metrics import pharmacodynamics`` import also works.
"""

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "..", "neuromod", "metrics", "pharmacodynamics.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("pharmacodynamics", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p = _load_module()


# ---------------------------------------------------------------------------------------
# Spatial variance / spectral scalars
# ---------------------------------------------------------------------------------------


def test_constant_plane_has_zero_variance():
    const = np.full((64, 64), 5.0)
    s = p.spectral_scalars(const)
    assert s["variance"] == pytest.approx(0.0, abs=1e-9)


def test_random_plane_has_higher_energy_than_constant():
    const = np.full((64, 64), 5.0)
    rng = np.random.RandomState(0)
    noisy = rng.randn(64, 64)
    assert p.spectral_scalars(noisy)["energy"] > p.spectral_scalars(const)["energy"]


def test_spectral_entropy_bounds_and_ordering():
    # White noise -> flat spectrum -> high entropy; a smooth ramp -> concentrated -> lower.
    rng = np.random.RandomState(1)
    noise = rng.rand(64, 64)
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    e_noise = p.spectral_scalars(noise)["spectral_entropy"]
    e_ramp = p.spectral_scalars(ramp)["spectral_entropy"]
    assert 0.0 <= e_ramp <= 1.0
    assert 0.0 <= e_noise <= 1.0
    assert e_noise > e_ramp


def test_high_low_ratio_tracks_high_frequency_content():
    # A high-frequency checkerboard should have a larger high/low band ratio than a smooth ramp.
    checproducer = np.indices((64, 64)).sum(axis=0) % 2
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))
    assert p.spectral_scalars(checproducer)["high_low_ratio"] > p.spectral_scalars(ramp)["high_low_ratio"]


# ---------------------------------------------------------------------------------------
# Latent metrics
# ---------------------------------------------------------------------------------------


def test_latent_metrics_shape_handling():
    latents = np.random.RandomState(2).randn(1, 4, 16, 16)  # [1, C, H, W]
    m = p.latent_spectral_metrics(latents)
    assert "latent_energy" in m and "latent_variance" in m
    assert "latent_ch0_energy" in m and "latent_ch3_energy" in m
    # Bare (C, H, W) should also work.
    m2 = p.latent_spectral_metrics(latents[0])
    assert m2["latent_variance"] == pytest.approx(m["latent_variance"], rel=1e-9)


def test_latent_variance_scales_with_amplitude():
    base = np.random.RandomState(3).randn(4, 16, 16)
    quiet = p.latent_spectral_metrics(base * 0.1)["latent_variance"]
    loud = p.latent_spectral_metrics(base * 10.0)["latent_variance"]
    assert loud > quiet


# ---------------------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------------------


def test_ssim_identical_is_one():
    img = (np.random.RandomState(4).rand(48, 48) * 255).astype(np.float64)
    assert p.ssim(img, img.copy()) == pytest.approx(1.0, abs=1e-6)


def test_ssim_decreases_with_noise():
    img = (np.random.RandomState(5).rand(48, 48) * 255).astype(np.float64)
    noisy = np.clip(img + np.random.RandomState(6).randn(48, 48) * 60, 0, 255)
    assert p.ssim(img, noisy) < 1.0


def test_ssim_shape_mismatch_raises():
    a = np.zeros((32, 32))
    b = np.zeros((16, 16))
    with pytest.raises(ValueError):
        p.ssim(a, b)


# ---------------------------------------------------------------------------------------
# Inter-seed diversity (mode-collapse metric) -- fallback paths
# ---------------------------------------------------------------------------------------


def test_diversity_zero_for_identical_images():
    img = (np.random.RandomState(7).rand(32, 32) * 255).astype(np.float64)
    imgs = [img.copy() for _ in range(5)]
    assert p.pairwise_diversity(imgs, method="ssim") == pytest.approx(0.0, abs=1e-6)


def test_diversity_positive_for_varied_images():
    rng = np.random.RandomState(8)
    imgs = [(rng.rand(32, 32) * 255).astype(np.float64) for _ in range(5)]
    assert p.pairwise_diversity(imgs, method="ssim") > 0.0


def test_diversity_collapse_ordering():
    # A near-identical "mode-collapsed" set should be less diverse than a varied set.
    rng = np.random.RandomState(9)
    base = (rng.rand(32, 32) * 255).astype(np.float64)
    collapsed = [np.clip(base + rng.randn(32, 32) * 1.0, 0, 255) for _ in range(5)]
    varied = [(rng.rand(32, 32) * 255).astype(np.float64) for _ in range(5)]
    assert p.pairwise_diversity(collapsed, method="l2") < p.pairwise_diversity(varied, method="l2")


def test_diversity_single_image_is_zero():
    img = np.zeros((16, 16))
    assert p.pairwise_diversity([img], method="ssim") == 0.0


# ---------------------------------------------------------------------------------------
# Per-generation bundle
# ---------------------------------------------------------------------------------------


def test_compute_image_metrics_always_has_pixel_scalars():
    img = (np.random.RandomState(10).rand(64, 64, 3) * 255).astype(np.uint8)
    m = p.compute_image_metrics(img)
    for key in ("pixel_energy", "pixel_variance", "pixel_high_low_ratio", "pixel_spectral_entropy"):
        assert key in m


def test_compute_image_metrics_with_latents_and_baseline():
    rng = np.random.RandomState(11)
    img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
    baseline = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
    latents = rng.randn(1, 4, 16, 16)
    m = p.compute_image_metrics(img, latents=latents, baseline_image=baseline)
    assert "latent_energy" in m
    assert "ssim_vs_baseline" in m
    # No CLIP/LPIPS backends in the test env -> those keys must be gracefully absent.
    if not p.clip_available():
        assert "clip_prompt_similarity" not in m
    if not p.lpips_available():
        assert "lpips_vs_baseline" not in m


def test_compute_image_metrics_ssim_baseline_identity():
    img = (np.random.RandomState(12).rand(64, 64, 3) * 255).astype(np.uint8)
    m = p.compute_image_metrics(img, baseline_image=img.copy())
    assert m["ssim_vs_baseline"] == pytest.approx(1.0, abs=1e-6)


def test_grayscale_conversion_from_rgba():
    rgba = (np.random.RandomState(13).rand(20, 20, 4) * 255).astype(np.uint8)
    gray = p.to_gray(rgba)
    assert gray.shape == (20, 20)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
