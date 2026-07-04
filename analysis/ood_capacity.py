#!/usr/bin/env python3
"""Thread C+ -- "Cocaine Crunch" as a capacity failure: over-stimulated models can't render OOD.

The reviewer's challenge: prove that stimulant "spectral constriction" is not just lower variance,
but an actual *loss of generative capacity* -- the over-dosed model becomes incapable of composing
novel / out-of-distribution scenes. We test it directly: sweep dose for an in-distribution prompt
("a tree") and for a battery of compositional OOD prompts, and compare how fast CLIP prompt-adherence
decays. If stimulants cause capacity collapse, adherence to the hard OOD prompts should fall faster
(and further) than to the easy in-distribution prompt.

Inputs are one long-format CSV per prompt (produced by ``demo/dose_response_runner.py``), since the
runner's ``clip_prompt_similarity`` is measured against that run's fixed ``--prompt``. This module
aggregates them: per (pack, prompt) it computes a **capacity-retention** ratio (top-dose adherence /
baseline adherence) and contrasts the OOD mean against the in-distribution reference.

Usage
-----
    python analysis/ood_capacity.py \
        --indist tree=outputs/ood/tree.csv \
        --ood "astronaut=outputs/ood/astronaut.csv,glass_snail=outputs/ood/snail.csv" \
        --packs cocaine,amphetamine --outdir outputs/ood/analysis
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))

# The in-distribution reference and a compositional / out-of-distribution battery. These are the
# prompts to run through demo/dose_response_runner.py (one CSV each) for a stimulant pack.
INDIST_PROMPT = "a tree"
OOD_PROMPTS = {
    "astronaut_horse": "an astronaut riding a horse on the moon",
    "glass_snail": "a snail made of translucent glass",
    "elephant_teapot": "a teapot shaped like an elephant",
    "clock_octopus": "an octopus juggling clocks underwater",
}

ADHERENCE = "clip_prompt_similarity"


def _load_stats():
    path = os.path.join(_HERE, "dose_response_stats.py")
    spec = importlib.util.spec_from_file_location("dose_response_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stats = _load_stats()


def retention(df: pd.DataFrame, pack: str, metric: str = ADHERENCE) -> Dict[str, float]:
    """Capacity-retention for one (pack) run: adherence at the top dose relative to baseline.

    Returns baseline / top-dose mean adherence and the retention ratio (1.0 = fully retained,
    ->0 = the model can no longer render the prompt at high dose). Uses the seed-level means.
    """
    sub = _stats._seed_level(df[(df["pack"] == pack) & (df["metric"] == metric)])
    if sub.empty:
        return {"baseline": np.nan, "topdose": np.nan, "retention": np.nan, "n_doses": 0}
    by_dose = sub.groupby("intensity")["value"].mean().sort_index()
    base = float(by_dose.iloc[0])
    top = float(by_dose.iloc[-1])
    ret = float(top / base) if base not in (0.0, np.nan) and np.isfinite(base) and base != 0 else np.nan
    return {"baseline": base, "topdose": top, "retention": ret, "n_doses": int(by_dose.size)}


def analyze(indist: Tuple[str, str], ood: Dict[str, str], packs: List[str]) -> pd.DataFrame:
    """Build the per-(pack, prompt) retention table + the OOD-vs-in-distribution capacity gap.

    ``indist`` is a (label, csv_path); ``ood`` maps label -> csv_path.
    """
    indist_label, indist_csv = indist
    frames = {indist_label: (_stats.load_long(indist_csv), "in-dist")}
    for label, path in ood.items():
        frames[label] = (_stats.load_long(path), "ood")

    rows = []
    for pack in packs:
        for label, (df, kind) in frames.items():
            r = retention(df, pack)
            rows.append({"pack": pack, "prompt": label, "prompt_class": kind, **r})
    table = pd.DataFrame(rows)

    # Capacity gap: mean OOD retention minus in-distribution retention, per pack.
    gaps = []
    for pack in packs:
        pt = table[table["pack"] == pack]
        ind = pt[pt["prompt_class"] == "in-dist"]["retention"].mean()
        ood_ret = pt[pt["prompt_class"] == "ood"]["retention"]
        gaps.append({"pack": pack, "indist_retention": ind,
                     "ood_retention_mean": float(ood_ret.mean()) if len(ood_ret) else np.nan,
                     "capacity_gap": float(ood_ret.mean() - ind) if len(ood_ret) else np.nan,
                     "n_ood_prompts": int(ood_ret.notna().sum())})
    return table, pd.DataFrame(gaps)


def plot(table: pd.DataFrame, gaps: pd.DataFrame, out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    packs = list(gaps["pack"])
    x = np.arange(len(packs))
    ax.bar(x - 0.2, gaps["indist_retention"], 0.4, label="in-distribution ('a tree')")
    ax.bar(x + 0.2, gaps["ood_retention_mean"], 0.4, label="OOD (compositional)")
    ax.set_xticks(x); ax.set_xticklabels(packs)
    ax.set_ylabel("Adherence retention (top dose / baseline)")
    ax.set_title("Capacity collapse: OOD prompts degrade faster under stimulant dose")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def _parse_named(s: str) -> Dict[str, str]:
    out = {}
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        label, _, path = part.partition("=")
        out[label.strip()] = path.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="OOD capacity collapse under stimulant dose")
    ap.add_argument("--indist", required=True, help="label=path for the in-distribution prompt CSV")
    ap.add_argument("--ood", required=True, help="Comma list of label=path for OOD prompt CSVs")
    ap.add_argument("--packs", default="cocaine,amphetamine")
    ap.add_argument("--outdir", default="outputs/ood/analysis")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    indist = _parse_named(args.indist)
    (il, ip), = indist.items()
    ood = _parse_named(args.ood)
    packs = [p.strip() for p in args.packs.split(",") if p.strip()]

    os.makedirs(args.outdir, exist_ok=True)
    table, gaps = analyze((il, ip), ood, packs)
    table.to_csv(os.path.join(args.outdir, "ood_retention.csv"), index=False)
    gaps.to_csv(os.path.join(args.outdir, "ood_capacity_gap.csv"), index=False)
    if not args.no_plots:
        plot(table, gaps, os.path.join(args.outdir, "ood_capacity.png"))
    print("OOD capacity gap (negative = OOD collapses more than in-distribution):")
    print(gaps.to_string(index=False))
    return gaps


if __name__ == "__main__":
    main()
