#!/usr/bin/env python3
"""
Thread C -- "Cocaine Crunch": mode collapse & spectral constriction (issue #9).

Turns the paper's static cocaine-vs-amphetamine snapshot (Table 2) into continuous
dose-response evidence that stimulants cause **spectral constriction -> mode collapse**:

  * **Inter-seed diversity collapse** -- mean pairwise LPIPS across seeds at each dose
    (already emitted by ``demo/dose_response_runner.py`` as ``interseed_diversity``).
    Collapse == diversity falls with dose.
  * **Constriction curves** -- latent spatial variance and spectral energy vs dose.
  * **The stimulant split** -- Cocaine (constriction: low variance AND low energy) vs
    Amphetamine (agitation: high variance, jittery) as diverging trajectories.
  * **OOD capacity** (optional) -- if the CSV carries ``clip_prompt_similarity`` for a
    creative prompt, plot how the over-dosed model drifts from an out-of-distribution target.

Consumes the tidy long-format CSV from the dose-response runner, so the statistics run
without a GPU. The seed-montage "collapse grid" requires saved images and is produced at
generation time (see :func:`collapse_grid_from_images`).

Usage
-----
    python analysis/mode_collapse.py --in outputs/dose_response/full.csv \
        --stimulants cocaine,amphetamine,methylphenidate --outdir outputs/dose_response/mode_collapse
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def _load_stats():
    """Reuse the dose-response stats layer (load_long, dose_curves, mann_kendall)."""
    path = os.path.join(_HERE, "dose_response_stats.py")
    spec = importlib.util.spec_from_file_location("dose_response_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stats = _load_stats()


# --------------------------------------------------------------------------------------
# Curves
# --------------------------------------------------------------------------------------


def diversity_curve(df: pd.DataFrame, pack: str) -> pd.DataFrame:
    """Inter-seed diversity vs dose for one pack (the mode-collapse signal).

    Reads the aggregate ``interseed_diversity`` rows (seed == 'ALL') the runner writes.
    """
    sub = df[(df["pack"] == pack) & (df["metric"] == "interseed_diversity")]
    out = (sub.groupby("intensity")["value"].mean().reset_index()
           .rename(columns={"value": "diversity"}).sort_values("intensity"))
    return out.reset_index(drop=True)


def constriction_curves(df: pd.DataFrame, pack: str,
                        metrics=("latent_variance", "latent_energy", "latent_high_low_ratio")) -> pd.DataFrame:
    """Per-dose mean + bootstrap CI for the constriction metrics of one pack."""
    packdf = df[df["pack"] == pack]
    curves = _stats.dose_curves(packdf, n_boot=2000)
    return curves[curves["metric"].isin(metrics)].reset_index(drop=True)


def classify_stimulant(df: pd.DataFrame, pack: str) -> Dict[str, object]:
    """Classify a stimulant's high-dose state as constriction vs agitation vs neither.

    Compares the top-dose mean of latent variance and energy against the sober (dose-0)
    baseline for the same pack:
      * both variance and energy fall  -> "constriction" (Cocaine-like lock-in)
      * variance stays high / rises     -> "agitation"   (Amphetamine-like jitter)
    """
    packdf = _stats._seed_level(df[df["pack"] == pack])
    doses = sorted(packdf["intensity"].unique())
    if len(doses) < 2:
        return {"pack": pack, "phenotype": "insufficient", "d_variance": np.nan, "d_energy": np.nan,
                "d_diversity": np.nan}
    lo, hi = doses[0], doses[-1]

    def _mean(metric, dose):
        v = packdf[(packdf["metric"] == metric) & (packdf["intensity"] == dose)]["value"]
        return float(v.mean()) if len(v) else np.nan

    d_var = _mean("latent_variance", hi) - _mean("latent_variance", lo)
    d_energy = _mean("latent_energy", hi) - _mean("latent_energy", lo)

    div = diversity_curve(df, pack)
    d_div = np.nan
    if len(div) >= 2:
        d_div = float(div.iloc[-1]["diversity"] - div.iloc[0]["diversity"])

    if np.isnan(d_var) or np.isnan(d_energy):
        phenotype = "unknown"
    elif d_var < 0 and d_energy < 0:
        phenotype = "constriction"       # cocaine lock-in
    elif d_var > 0:
        phenotype = "agitation"          # amphetamine jitter
    else:
        phenotype = "mixed"
    return {"pack": pack, "phenotype": phenotype, "d_variance": d_var, "d_energy": d_energy,
            "d_diversity": d_div}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def plot_diversity_collapse(df: pd.DataFrame, packs: List[str], out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for pack in packs:
        c = diversity_curve(df, pack)
        if len(c):
            ax.plot(c["intensity"], c["diversity"], marker="o", label=pack)
    ax.set_xlabel("Dose (intensity)")
    ax.set_ylabel("Inter-seed diversity (mean pairwise LPIPS)")
    ax.set_title("Mode-collapse: output diversity vs dose")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def collapse_grid_from_images(images_by_seed: Dict[int, "object"], out_path: str,
                              max_seeds: int = 16) -> bool:
    """Build a montage of N seeds at a fixed high dose (visual proof of collapse).

    Requires already-generated PIL images (produced at GPU generation time). No-op-safe:
    returns False if matplotlib/images are unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    seeds = sorted(images_by_seed)[:max_seeds]
    if not seeds:
        return False
    cols = int(np.ceil(np.sqrt(len(seeds))))
    rows = int(np.ceil(len(seeds) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, seed in zip(axs, seeds):
        ax.imshow(images_by_seed[seed])
        ax.set_title(f"seed {seed}", fontsize=6)
    fig.suptitle("Seed diversity at high stimulant dose (collapse grid)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def analyze(csv_path: str, outdir: str, stimulants: List[str], plots: bool = True) -> pd.DataFrame:
    os.makedirs(outdir, exist_ok=True)
    df = _stats.load_long(csv_path)
    present = [p for p in stimulants if p in set(df["pack"].unique())]

    rows = [classify_stimulant(df, p) for p in present]
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(outdir, "stimulant_phenotypes.csv"), index=False)

    # Per-pack constriction curves (dose_curves already carries the 'pack' column).
    all_curves = [constriction_curves(df, p) for p in present]
    all_curves = [c for c in all_curves if len(c)]
    if all_curves:
        pd.concat(all_curves).to_csv(os.path.join(outdir, "constriction_curves.csv"), index=False)

    if plots and present:
        plot_diversity_collapse(df, present, os.path.join(outdir, "diversity_collapse.png"))

    if not summary.empty:
        print("Stimulant phenotypes (high dose vs sober):")
        print(summary.to_string(index=False))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cocaine Crunch: stimulant mode-collapse analysis")
    ap.add_argument("--in", dest="csv_path", required=True)
    ap.add_argument("--outdir", default="outputs/dose_response/mode_collapse")
    ap.add_argument("--stimulants", default="cocaine,amphetamine,methylphenidate")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    stimulants = [s.strip() for s in args.stimulants.split(",") if s.strip()]
    analyze(args.csv_path, args.outdir, stimulants, plots=not args.no_plots)


if __name__ == "__main__":
    main()
