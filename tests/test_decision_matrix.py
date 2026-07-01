"""Unit tests for the pilot decision matrix (issue #11). Pure scoring, no GPU."""

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        "decision_matrix", os.path.join(_HERE, "..", "analysis", "decision_matrix.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dm = _load()


def test_stronger_stat_signal_ranks_higher_all_else_equal():
    # Same thread priors would confound; use two threads with identical priors override.
    signals = [
        {"thread": "mode_collapse", "stat_strength_raw": 10.0, "visual_drama_raw": 1.0,
         "security_relevance": 0.7, "novelty": 0.6},
        {"thread": "mode_collapse", "stat_strength_raw": 0.1, "visual_drama_raw": 1.0,
         "security_relevance": 0.7, "novelty": 0.6},
    ]
    df = dm.score_threads(signals)
    assert df.iloc[0]["stat_strength_raw"] == 10.0
    assert df.iloc[0]["statistical_strength"] == pytest.approx(1.0)
    assert df.iloc[1]["statistical_strength"] == pytest.approx(0.0)


def test_priors_break_ties_when_data_equal():
    # Equal data signals -> normalized to 0.5 each; security/novelty priors decide.
    signals = [
        {"thread": "safety_boundary", "stat_strength_raw": 1.0, "visual_drama_raw": 1.0},
        {"thread": "mode_collapse", "stat_strength_raw": 1.0, "visual_drama_raw": 1.0},
    ]
    rec = dm.recommend(signals)
    # safety_boundary has higher security+novelty priors -> should win the tie.
    assert rec["headline"] == "safety_boundary"


def test_nan_stat_normalizes_to_zero():
    signals = [
        {"thread": "latent_specter", "stat_strength_raw": float("nan"), "visual_drama_raw": 5.0},
        {"thread": "mode_collapse", "stat_strength_raw": 2.0, "visual_drama_raw": 1.0},
    ]
    df = dm.score_threads(signals)
    specter = df[df["thread"] == "latent_specter"].iloc[0]
    assert specter["statistical_strength"] == 0.0


def test_weights_sum_and_total_bounds():
    assert abs(sum(dm.DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
    signals = [{"thread": "vitals_monitor", "stat_strength_raw": 1.0, "visual_drama_raw": 1.0}]
    df = dm.score_threads(signals)
    assert 0.0 <= df.iloc[0]["total"] <= 1.0


def test_empty_input():
    assert dm.score_threads([]).empty
    assert dm.recommend([])["headline"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
