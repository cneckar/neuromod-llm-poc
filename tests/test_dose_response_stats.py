"""
Unit tests for the dose-response statistics layer.

Loaded by file path to avoid the torch-importing ``neuromod`` package __init__; the module
itself depends only on numpy/pandas/scipy.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULE_PATH = os.path.join(_HERE, "..", "analysis", "dose_response_stats.py")


def _load():
    spec = importlib.util.spec_from_file_location("dose_response_stats", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s = _load()


def test_bootstrap_ci_brackets_mean():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = bootstrap = s.bootstrap_ci(vals, n_boot=2000)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["mean"] == pytest.approx(3.0)
    assert ci["n"] == 5


def test_bootstrap_ci_single_value():
    ci = s.bootstrap_ci(np.array([7.0]))
    assert ci["mean"] == ci["ci_low"] == ci["ci_high"] == 7.0


def test_mann_kendall_detects_monotonic_increase():
    mk = s.mann_kendall(np.arange(11).astype(float))
    assert mk["trend"] == "increasing"
    assert mk["p"] < 0.05


def test_mann_kendall_flat_is_no_trend():
    mk = s.mann_kendall(np.ones(11))
    assert mk["trend"] == "none"


def test_breakpoint_finds_the_cliff():
    doses = np.linspace(0, 1, 11)
    means = np.where(doses < 0.6, 1.0, 0.0)  # sharp cliff between 0.5 and 0.6
    bp = s.detect_breakpoint(doses, means)
    assert bp["breakpoint_dose"] == pytest.approx(0.55, abs=0.051)
    assert bp["sharpness"] == pytest.approx(1.0, abs=1e-6)


def test_benjamini_hochberg_monotone_and_bounded():
    q = s.benjamini_hochberg([0.01, 0.02, 0.03, 0.5])
    assert all(0.0 <= x <= 1.0 for x in q)
    assert q[0] <= q[-1]


def test_dose_curves_and_trends_end_to_end():
    # Build a synthetic long-format frame: metric increases with dose.
    rows = []
    rng = np.random.RandomState(0)
    for dose in [round(0.1 * i, 1) for i in range(11)]:
        for seed in range(8):
            rows.append({"pack": "lsd", "intensity": dose, "seed": seed,
                         "metric": "clip_prompt_similarity", "value": 1.0 - dose + rng.randn() * 0.01})
    df = pd.DataFrame(rows)
    curves = s.dose_curves(df, n_boot=500)
    assert set(curves["intensity"]) == {round(0.1 * i, 1) for i in range(11)}
    trends = s.trend_summary(curves)
    row = trends[trends["metric"] == "clip_prompt_similarity"].iloc[0]
    assert row["trend"] == "decreasing"
    assert row["spearman_rho"] < -0.9


def test_load_long_dedupes_resumed_rows(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(
        "pack,intensity,seed,metric,value\n"
        "lsd,0.0,ALL,interseed_diversity,0.1\n"
        "lsd,0.0,ALL,interseed_diversity,0.2\n"  # resumed re-append; last should win
    )
    df = s.load_long(str(csv))
    assert len(df) == 1
    assert df.iloc[0]["value"] == 0.2


def test_cohens_d_paired_large_effect():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treat = np.array([3.0, 5.0, 7.0, 9.0, 11.0])
    d = s.cohens_d_paired(base, treat)
    assert d > 1.0  # clearly large positive effect


def test_cohens_d_paired_insufficient():
    assert np.isnan(s.cohens_d_paired(np.array([1.0]), np.array([2.0])))


def test_fit_dose_response_curve_recovers_sigmoid():
    fitter_available = s._ablations_fitter() is not None
    if not fitter_available:
        pytest.skip("ablations_analysis fitter unavailable in this environment")
    doses = np.linspace(0, 1, 11)
    means = 1.0 / (1.0 + np.exp(-15.0 * (doses - 0.5)))  # steep sigmoid, EC50=0.5
    out = s.fit_dose_response_curve(doses, means)
    assert out["curve_r2"] > 0.9
    # When the sigmoid model wins, EC50 must land near the true center.
    if out["curve_type"] == "sigmoid":
        assert 0.3 < out["ec50"] < 0.7


def test_fit_dose_response_curve_flat_returns_empty():
    doses = np.linspace(0, 1, 11)
    means = np.ones(11)
    out = s.fit_dose_response_curve(doses, means)
    assert np.isnan(out["ec50"]) and out["curve_type"] is None


def test_effect_size_summary_end_to_end():
    if s._cliffs_delta_fn() is None:
        pytest.skip("statistical_analysis Cliff's delta unavailable in this environment")
    rng = np.random.RandomState(0)
    rows = []
    for dose in [0.0, 0.5, 1.0]:
        shift = dose * 5.0
        # Enough seeds that std(diff) is stable, so the larger dose has the larger
        # standardized (paired-d) effect rather than it being noise-dominated.
        for seed in range(40):
            rows.append({"pack": "lsd", "intensity": dose, "seed": seed,
                         "metric": "clip_prompt_similarity", "value": shift + rng.randn() * 0.1})
    df = pd.DataFrame(rows)
    es = s.effect_size_summary(df)
    row = es[es["metric"] == "clip_prompt_similarity"].iloc[0]
    assert abs(row["cohens_d"]) > 2.0          # strong separation from baseline
    assert abs(row["cliffs_delta"]) > 0.9       # near-complete dominance
    assert row["dose_of_max"] == 1.0            # largest dose = largest effect


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
