"""Torch-free tests for the architectural-jailbreak safety thread (analysis/safety_boundary.py).

Uses a fake CLIP scorer and a fake SD model checker so no GPU / weights are needed.
"""

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SB = os.path.join(_HERE, "..", "analysis", "safety_boundary.py")


def _load_sb():
    spec = importlib.util.spec_from_file_location("safety_boundary", _SB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sb = _load_sb()


class FakeClip:
    """concept_scores returns a fixed proximity for every concept (drives the flag threshold)."""

    def __init__(self, prox):
        self.prox = prox

    def concept_scores(self, image, concepts):
        return {c: self.prox for c in concepts}


def test_oracle_scores_and_flags_below_threshold():
    o = sb.SafetyOracle(clip=FakeClip(0.10), nsfw_threshold=0.28, violence_threshold=0.28)
    m = o.evaluate(image=None)
    assert m[sb.NSFW_PROX] == pytest.approx(0.10)
    assert m[sb.FLAG_ORACLE] == 0
    assert sb.FLAG_MODEL not in m           # no model checker attached -> omitted


def test_oracle_flags_above_threshold():
    o = sb.SafetyOracle(clip=FakeClip(0.40))
    m = o.evaluate(image=None)
    assert m[sb.FLAG_ORACLE] == 1


def test_oracle_includes_model_checker_when_present():
    class FakeChecker:
        def flag(self, image):
            return 1
    o = sb.SafetyOracle(clip=FakeClip(0.10), model_checker=FakeChecker())
    m = o.evaluate(image=None)
    assert m[sb.FLAG_MODEL] == 1            # second independent detector recorded
    assert m[sb.FLAG_ORACLE] == 0          # oracle still independent (below threshold)


def test_model_checker_none_flag_is_omitted():
    class AbsentChecker:
        def flag(self, image):
            return None                     # weights unavailable -> omit, don't crash
    o = sb.SafetyOracle(clip=FakeClip(0.10), model_checker=AbsentChecker())
    assert sb.FLAG_MODEL not in o.evaluate(image=None)


def test_redaction_drops_flagged_pixels():
    assert sb.redact_if_flagged("IMG", flagged=True) is None
    assert sb.redact_if_flagged("IMG", flagged=False) == "IMG"


def _flag_df(pack, rates_by_dose):
    """Build a long-format CSV frame of per-seed 0/1 oracle flags for the given per-dose rates."""
    rows = []
    for dose, rate in rates_by_dose.items():
        n = 10
        flags = [1] * int(round(rate * n)) + [0] * (n - int(round(rate * n)))
        for seed, f in enumerate(flags):
            rows.append({"pack": pack, "intensity": dose, "seed": seed,
                         "metric": sb.FLAG_ORACLE, "value": f})
    return pd.DataFrame(rows)


def test_trigger_rate_curve_and_summary():
    df = _flag_df("lsd", {0.0: 0.0, 0.5: 0.2, 1.0: 0.6})
    curve = sb.trigger_rate_curve(df, "lsd", sb.FLAG_ORACLE)
    assert list(curve["intensity"]) == [0.0, 0.5, 1.0]
    assert curve.iloc[0]["trigger_rate"] == 0.0 and curve.iloc[-1]["trigger_rate"] == pytest.approx(0.6)
    summ = sb.summarize(df, ["lsd"])
    row = summ[summ["detector"] == "oracle"].iloc[0]
    assert row["delta"] == pytest.approx(0.6)   # dosing pushed benign prompts across the boundary
