"""Torch-free test for the OOD-capacity contrast (analysis/ood_capacity.py)."""

import importlib.util
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    path = os.path.join(_HERE, "..", "analysis", "ood_capacity.py")
    spec = importlib.util.spec_from_file_location("ood_capacity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


oc = _load()


def _adherence_csv(path, pack, base, top, n=8):
    """Write a long CSV where clip_prompt_similarity goes base->top linearly over 3 doses."""
    rows = []
    for dose, val in ((0.0, base), (0.5, (base + top) / 2), (1.0, top)):
        for seed in range(n):
            rows.append({"pack": pack, "intensity": dose, "seed": seed,
                         "metric": "clip_prompt_similarity", "value": val})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ood_collapses_more_than_indist(tmp_path):
    ind = str(tmp_path / "tree.csv")
    ood = str(tmp_path / "astro.csv")
    # In-distribution "a tree": adherence barely drops (0.30 -> 0.27, retention ~0.9).
    _adherence_csv(ind, "cocaine", base=0.30, top=0.27)
    # OOD compositional prompt: adherence craters (0.30 -> 0.09, retention ~0.3).
    _adherence_csv(ood, "cocaine", base=0.30, top=0.09)

    table, gaps = oc.analyze(("tree", ind), {"astro": ood}, ["cocaine"])
    ind_ret = table[(table.prompt == "tree")].iloc[0]["retention"]
    ood_ret = table[(table.prompt == "astro")].iloc[0]["retention"]
    assert ind_ret > 0.8 and ood_ret < 0.5           # OOD retains far less capacity
    g = gaps.iloc[0]
    assert g["capacity_gap"] < 0                      # OOD retention below in-distribution
    assert g["n_ood_prompts"] == 1


def test_parse_named_pairs():
    assert oc._parse_named("a=x.csv, b=y.csv") == {"a": "x.csv", "b": "y.csv"}
