"""
GPU-free unit tests for the four dose-response "hero" thread scripts (issues #7-#10).

Each script is loaded by file path (the analysis/ and demo/ dirs are not packages) and
exercised on synthetic long-format data or synthetic frames, so the analysis/statistics and
the exporters are verified without a GPU or real generations.
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


mode_collapse = _load("mode_collapse", "analysis/mode_collapse.py")
latent_specter = _load("latent_specter", "analysis/latent_specter.py")
safety_boundary = _load("safety_boundary", "analysis/safety_boundary.py")
vitals = _load("vitals_monitor", "demo/vitals_monitor.py")


def _long_rows(pack, metric, per_dose_fn, seeds=8, doses=None):
    doses = doses or [round(0.1 * i, 1) for i in range(11)]
    rng = np.random.RandomState(abs(hash((pack, metric))) % 2**31)
    rows = []
    for d in doses:
        for s in range(seeds):
            rows.append({"pack": pack, "intensity": d, "seed": s, "metric": metric,
                         "value": per_dose_fn(d) + rng.randn() * 0.01})
    return rows


# ------------------------------------------------------------------ Thread C: mode collapse


def _mode_collapse_df():
    rows = []
    # Cocaine: constriction -> variance and energy fall, diversity collapses.
    rows += _long_rows("cocaine", "latent_variance", lambda d: 700 - 250 * d)
    rows += _long_rows("cocaine", "latent_energy", lambda d: 110 - 12 * d)
    # Amphetamine: agitation -> variance rises.
    rows += _long_rows("amphetamine", "latent_variance", lambda d: 700 + 120 * d)
    rows += _long_rows("amphetamine", "latent_energy", lambda d: 110 + 4 * d)
    # Inter-seed diversity (aggregate ALL rows).
    for d in [round(0.1 * i, 1) for i in range(11)]:
        rows.append({"pack": "cocaine", "intensity": d, "seed": "ALL",
                     "metric": "interseed_diversity", "value": 0.9 - 0.6 * d})
        rows.append({"pack": "amphetamine", "intensity": d, "seed": "ALL",
                     "metric": "interseed_diversity", "value": 0.85})
    return pd.DataFrame(rows)


def test_mode_collapse_diversity_curve_falls():
    df = _mode_collapse_df()
    c = mode_collapse.diversity_curve(df, "cocaine")
    assert len(c) == 11
    assert c.iloc[-1]["diversity"] < c.iloc[0]["diversity"]  # collapse


def test_mode_collapse_classifies_constriction_vs_agitation():
    df = _mode_collapse_df()
    coke = mode_collapse.classify_stimulant(df, "cocaine")
    speed = mode_collapse.classify_stimulant(df, "amphetamine")
    assert coke["phenotype"] == "constriction" and coke["d_variance"] < 0 and coke["d_energy"] < 0
    assert speed["phenotype"] == "agitation" and speed["d_variance"] > 0


def test_mode_collapse_analyze_writes_summary(tmp_path):
    df = _mode_collapse_df()
    csv = tmp_path / "mc.csv"
    df.to_csv(csv, index=False)
    summary = mode_collapse.analyze(str(csv), str(tmp_path / "out"), ["cocaine", "amphetamine"],
                                    plots=False)
    assert set(summary["phenotype"]) == {"constriction", "agitation"}
    assert (tmp_path / "out" / "stimulant_phenotypes.csv").exists()


# ------------------------------------------------------------------ Thread A: latent specter

_CONCEPT = "a human figure"


def _specter_df():
    cm = latent_specter.concept_metric(_CONCEPT)
    rows = []
    # DMT: ghost concept rises with dose; prompt stays anchored.
    rows += _long_rows("dmt", cm, lambda d: 0.12 + 0.25 * d)
    rows += _long_rows("dmt", "clip_prompt_similarity", lambda d: 0.30)
    # Placebo (random vector): no ghost.
    rows += _long_rows("placebo", cm, lambda d: 0.11)
    rows += _long_rows("placebo", "clip_prompt_similarity", lambda d: 0.30)
    return pd.DataFrame(rows)


def test_specter_ghost_curve_rises():
    c = latent_specter.ghost_curve(_specter_df(), "dmt", _CONCEPT)
    assert c.iloc[-1]["concept_proximity"] > c.iloc[0]["concept_proximity"]
    # prompt anchor stays roughly flat/high
    assert abs(c.iloc[-1]["prompt_similarity"] - c.iloc[0]["prompt_similarity"]) < 0.05


def test_specter_placebo_contrast_is_large():
    if latent_specter._stats._cliffs_delta_fn() is None:
        pytest.skip("cliffs delta unavailable")
    con = latent_specter.placebo_contrast(_specter_df(), "dmt", "placebo", _CONCEPT)
    assert con["mean_treatment"] > con["mean_placebo"]
    assert con["cohens_d"] > 2.0
    assert con["cliffs_delta"] > 0.9


def test_specter_pareidolia_fp_rate_low():
    fp = latent_specter.pareidolia_fp_rate(_specter_df(), ["placebo"], _CONCEPT, threshold=0.25)
    assert fp["fp_rate"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------------ Thread B: safety boundary


class _FakeCLIP:
    def __init__(self, proximity):
        self.proximity = proximity

    def concept_scores(self, image, concepts):
        return {c: self.proximity for c in concepts}


def test_safety_oracle_flag_threshold():
    hot = safety_boundary.SafetyOracle(clip=_FakeCLIP(0.4))
    cold = safety_boundary.SafetyOracle(clip=_FakeCLIP(0.1))
    assert hot.flag(object()) == 1
    assert cold.flag(object()) == 0
    s = hot.score(object())
    assert s[safety_boundary.NSFW_PROX] == pytest.approx(0.4)


def test_safety_redaction_drops_flagged_pixels():
    img = object()
    assert safety_boundary.redact_if_flagged(img, True) is None
    assert safety_boundary.redact_if_flagged(img, False) is img


def test_safety_trigger_rate_rises_with_dose():
    rows = []
    for d in [round(0.1 * i, 1) for i in range(11)]:
        for s in range(10):
            flag = 1 if (d >= 0.7 and s < int(10 * (d - 0.6))) else 0
            rows.append({"pack": "lsd", "intensity": d, "seed": s,
                         "metric": safety_boundary.FLAG_ORACLE, "value": flag})
    df = pd.DataFrame(rows)
    c = safety_boundary.trigger_rate_curve(df, "lsd", safety_boundary.FLAG_ORACLE)
    assert c.iloc[-1]["trigger_rate"] > c.iloc[0]["trigger_rate"]
    summ = safety_boundary.summarize(df, ["lsd"])
    assert summ.iloc[0]["delta"] > 0


# ------------------------------------------------------------------ Thread D: vitals monitor


def test_vitals_synthetic_frames_shape():
    frames = vitals.synthetic_frames(n=8, size=48)
    assert len(frames) == 8
    assert frames[0].dose == 0.0 and frames[-1].dose == 1.0
    # CLIP anchor decays, energy rises across the sweep.
    assert frames[-1].metrics["clip_prompt_similarity"] < frames[0].metrics["clip_prompt_similarity"]
    assert frames[-1].metrics["latent_energy"] > frames[0].metrics["latent_energy"]


def test_vitals_render_composite_returns_image():
    frames = vitals.synthetic_frames(n=5, size=48)
    metrics = vitals._default_metrics(frames)
    img = vitals.render_composite(frames, 2, metrics)
    assert hasattr(img, "size") and img.size[0] > 0


def test_vitals_build_all_writes_gif_and_html(tmp_path):
    frames = vitals.synthetic_frames(n=6, size=48)
    out = vitals.build_all(frames, str(tmp_path), "Test PD")
    assert os.path.exists(out["gif"]) and os.path.getsize(out["gif"]) > 0
    assert os.path.exists(out["html"])
    html = open(out["html"]).read()
    assert "data:image/png;base64" in html and "Test PD" in html


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
