"""
GPU-free tests for the dose-response runner's image-saving + safety-oracle wiring and the
pilot's monotonicity-based statistical-strength scoring.

All modules are loaded by file path (demo/, analysis/, scripts/ are not packages) and driven
through the synthetic DryRunGenerator + a fake CLIP, so no torch/GPU/model weights are needed.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("dose_response_runner", "demo/dose_response_runner.py")
run_pilot = _load("run_pilot", "scripts/run_pilot.py")
safety_boundary = _load("safety_boundary", "analysis/safety_boundary.py")


# --------------------------------------------------------------------- image save / load


def test_image_filename_is_deterministic_and_parseable():
    assert runner.image_filename("dmt", 0.5, 3) == "dmt__i0.50__s3.png"
    assert runner.image_filename("dmt", 1.0, 12) == "dmt__i1.00__s12.png"


def test_save_and_load_images_roundtrip(tmp_path):
    gen = runner.DryRunGenerator("a tree", size=32)
    saved = {}
    for seed in range(4):
        img = gen.generate("cocaine", 1.0, seed)["image"]
        runner.save_image(img, str(tmp_path), "cocaine", 1.0, seed)
        saved[seed] = img
    loaded = runner.load_saved_images(str(tmp_path), "cocaine", 1.0)
    assert sorted(loaded) == [0, 1, 2, 3]
    # Pixels survive the PNG round-trip (uint8, lossless).
    assert np.array_equal(np.asarray(loaded[0]), saved[0])
    # A different dose loads nothing.
    assert runner.load_saved_images(str(tmp_path), "cocaine", 0.5) == {}


def test_run_saves_images_when_image_dir_set(tmp_path):
    gen = runner.DryRunGenerator("a tree", size=32)
    img_dir = tmp_path / "imgs"
    runner.run(gen, packs=["lsd"], intensities=[0.0, 0.5, 1.0], seeds=[0, 1],
               prompt="a tree", csv_path=str(tmp_path / "r.csv"),
               diversity_method="ssim", verbose=False, image_dir=str(img_dir))
    pngs = sorted(os.listdir(img_dir))
    assert len(pngs) == 6  # 3 doses x 2 seeds
    assert "lsd__i1.00__s0.png" in pngs


# --------------------------------------------------------------------- safety oracle wiring


class _FakeCLIP:
    """Returns a proximity that climbs with dose is impossible here (no dose context);
    instead a fixed proximity, high enough to trip the oracle for the 'hot' case."""

    def __init__(self, proximity):
        self.proximity = proximity

    def concept_scores(self, image, concepts):
        return {c: self.proximity for c in concepts}


def test_run_writes_oracle_metrics_and_redacts_flagged(tmp_path):
    gen = runner.DryRunGenerator("a tree", size=32)
    oracle = safety_boundary.SafetyOracle(clip=_FakeCLIP(0.9))  # always flags
    img_dir = tmp_path / "imgs"
    csv = tmp_path / "safety.csv"
    runner.run(gen, packs=["lsd"], intensities=[0.0, 1.0], seeds=[0, 1],
               prompt="a tree", csv_path=str(csv), diversity_method="ssim",
               verbose=False, image_dir=str(img_dir), safety_oracle=oracle)
    df = pd.read_csv(csv)
    metrics = set(df["metric"])
    assert safety_boundary.FLAG_ORACLE in metrics
    assert safety_boundary.NSFW_PROX in metrics
    # Every generation flagged -> all pixels redacted (nothing saved).
    assert not img_dir.exists() or os.listdir(img_dir) == []


def test_run_model_flag_recorded_when_generator_surfaces_it(tmp_path):
    class _FlaggingGen(runner.DryRunGenerator):
        def generate(self, pack, intensity, seed):
            r = super().generate(pack, intensity, seed)
            r[runner.SAFETY_FLAG_MODEL] = 1 if intensity >= 1.0 else 0
            return r

    gen = _FlaggingGen("a tree", size=32)
    oracle = safety_boundary.SafetyOracle(clip=_FakeCLIP(0.0))  # oracle never flags
    csv = tmp_path / "s.csv"
    runner.run(gen, packs=["lsd"], intensities=[0.0, 1.0], seeds=[0],
               prompt="a tree", csv_path=str(csv), diversity_method="ssim",
               verbose=False, safety_oracle=oracle)
    df = pd.read_csv(csv)
    model_rows = df[df["metric"] == runner.SAFETY_FLAG_MODEL]
    assert set(model_rows["intensity"]) == {0.0, 1.0}
    assert model_rows[model_rows["intensity"] == 1.0]["value"].iloc[0] == 1


# --------------------------------------------------------------------- monotonicity scoring


def _monotone_df(pack, metric, slope, seeds=8, noise=0.001):
    # Deterministic seed (not hash(), which varies with PYTHONHASHSEED) so the test is reproducible.
    seed = (sum(map(ord, pack + metric)) + int(abs(slope) * 100)) % 2**31
    rng = np.random.RandomState(seed)
    rows = []
    for d in [round(0.1 * i, 1) for i in range(11)]:
        for s in range(seeds):
            rows.append({"pack": pack, "intensity": d, "seed": s, "metric": metric,
                         "value": slope * d + rng.randn() * noise})
    return pd.DataFrame(rows)


def test_monotonicity_strength_prefers_strong_monotone_metric():
    # A near-perfectly monotone metric should score ~1.0.
    df = _monotone_df("dmt", "latent_energy", slope=5.0)
    strength, metric = run_pilot._monotonicity_strength(df, ["dmt"])
    assert metric == "latent_energy"
    assert strength == pytest.approx(1.0, abs=0.05)


def test_monotonicity_strength_low_for_flat_metric():
    # A truly flat metric (no dose response) has no defined rank trend -> strength ~ 0.
    df = _monotone_df("placebo", "latent_energy", slope=0.0, noise=0.0)
    strength, _ = run_pilot._monotonicity_strength(df, ["placebo"])
    assert strength < 0.5


def test_thread_signal_reports_driving_metric():
    df = pd.concat([
        _monotone_df("cocaine", "latent_variance", slope=-6.0),
        _monotone_df("cocaine", "latent_energy", slope=-0.5),
    ])
    cfg = {"key": "mode_collapse", "packs": ["cocaine"], "hero": "latent_variance"}
    sig = run_pilot._thread_signal(cfg, df)
    assert sig["thread"] == "mode_collapse"
    # The strongest monotone metric drives the statistical-strength score.
    assert sig["stat_strength_raw"] == pytest.approx(1.0, abs=0.05)
    assert sig["stat_metric"] in {"latent_variance", "latent_energy"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
