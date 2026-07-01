#!/usr/bin/env python3
"""
Thread A -- "Latent Specter": statistical ghost detection (issue #7).

Turns the single DMT "Latent Ghost" anecdote (paper fig:dmt_ghost) into a statistical
result across many generations: high-entropy packs (DMT/LSD) exhume off-prompt structural
priors -- e.g. a human figure -- that rise with dose, while the pixel output stays anchored
to the prompt ("a tree"), and this is **absent under a placebo (random-vector) pack**.

Evidence produced:
  * **Ghost prevalence vs dose** -- off-prompt concept proximity (from CLIP concept scoring
    of the decoded image) rises with dose while ``clip_prompt_similarity`` stays high
    (the "semantic dissociation").
  * **Placebo control** -- effect size of concept proximity at the top dose, treatment
    (DMT/LSD) vs placebo. A real ghost is a large treatment>>placebo gap, proving it is a
    training prior rather than random noise.
  * **Pareidolia / false-positive rate** -- fraction of null (baseline/placebo) generations
    whose concept proximity exceeds a *pre-registered* threshold.

Consumes the runner CSV; the off-prompt concept columns are produced by
``demo/dose_response_runner.py --concepts "a human figure,a face,..."`` (CLIP concept
scoring), so this analysis runs without a GPU. The exhumed-ghost montage is built from saved
images at generation time (:func:`ghost_montage_from_images`).

Usage
-----
    python analysis/latent_specter.py --in outputs/dose_response/full.csv \
        --treatments dmt,lsd --placebo placebo --concept "a human figure" \
        --outdir outputs/dose_response/latent_specter
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_stats():
    path = os.path.join(_HERE, "dose_response_stats.py")
    spec = importlib.util.spec_from_file_location("dose_response_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stats = _load_stats()

CONCEPT_PREFIX = "concept::"


def concept_metric(concept: str) -> str:
    """CSV metric name for an off-prompt concept (matches the runner's naming)."""
    return f"{CONCEPT_PREFIX}{concept}"


# --------------------------------------------------------------------------------------
# Curves & statistics
# --------------------------------------------------------------------------------------


def ghost_curve(df: pd.DataFrame, pack: str, concept: str,
                prompt_metric: str = "clip_prompt_similarity") -> pd.DataFrame:
    """Off-prompt concept proximity and prompt similarity vs dose for one pack.

    The ghost signature is: ``concept`` rises with dose while ``prompt_sim`` stays high
    (semantic dissociation). ``dissociation`` = concept - prompt_sim is a convenience column.
    """
    cmetric = concept_metric(concept)
    packdf = _stats._seed_level(df[df["pack"] == pack])
    if packdf.empty:
        return pd.DataFrame(columns=["intensity", "concept_proximity", "prompt_similarity",
                                     "dissociation"])
    rows = []
    for intensity, sub in packdf.groupby("intensity"):
        concept_vals = sub[sub["metric"] == cmetric]["value"]
        prompt_vals = sub[sub["metric"] == prompt_metric]["value"]
        rows.append({
            "intensity": float(intensity),
            "concept_proximity": float(concept_vals.mean()) if len(concept_vals) else np.nan,
            "prompt_similarity": float(prompt_vals.mean()) if len(prompt_vals) else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("intensity").reset_index(drop=True)
    out["dissociation"] = out["concept_proximity"] - out["prompt_similarity"]
    return out


def placebo_contrast(df: pd.DataFrame, treatment: str, placebo: str, concept: str,
                     dose: Optional[float] = None) -> Dict[str, float]:
    """Effect size of concept proximity, treatment vs placebo, at a fixed (top) dose.

    A large positive Cohen's d / Cliff's delta means the ghost is specific to the steered
    treatment, not random-vector noise -- the core control for this thread.
    """
    cmetric = concept_metric(concept)
    tdf = _stats._seed_level(df[df["pack"] == treatment])
    pdf = _stats._seed_level(df[df["pack"] == placebo])
    if dose is None:
        doses = tdf[tdf["metric"] == cmetric]["intensity"]
        dose = float(doses.max()) if len(doses) else np.nan
    treat = tdf[(tdf["metric"] == cmetric) & (tdf["intensity"] == dose)].sort_values("seed")["value"].values
    plac = pdf[(pdf["metric"] == cmetric) & (pdf["intensity"] == dose)].sort_values("seed")["value"].values
    cliffs = _stats._cliffs_delta_fn()
    return {
        "dose": dose,
        "n_treatment": int(treat.size),
        "n_placebo": int(plac.size),
        "mean_treatment": float(np.mean(treat)) if treat.size else np.nan,
        "mean_placebo": float(np.mean(plac)) if plac.size else np.nan,
        "cohens_d": _stats.cohens_d_paired(plac, treat) if min(treat.size, plac.size) >= 2 else np.nan,
        "cliffs_delta": float(cliffs(plac, treat)) if (cliffs and treat.size and plac.size) else np.nan,
    }


def pareidolia_fp_rate(df: pd.DataFrame, null_packs: List[str], concept: str,
                       threshold: float) -> Dict[str, float]:
    """False-positive rate: fraction of null generations whose concept proximity exceeds a
    pre-registered ``threshold`` (the ghost detector's pareidolia floor)."""
    cmetric = concept_metric(concept)
    sub = _stats._seed_level(df[df["pack"].isin(null_packs) & (df["metric"] == cmetric)])
    vals = sub["value"].values
    if vals.size == 0:
        return {"threshold": threshold, "n": 0, "fp_rate": np.nan}
    return {"threshold": threshold, "n": int(vals.size),
            "fp_rate": float(np.mean(vals > threshold))}


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def plot_ghost_prevalence(df: pd.DataFrame, treatments: List[str], placebo: Optional[str],
                          concept: str, out_path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    for pack in treatments:
        c = ghost_curve(df, pack, concept)
        ax.plot(c["intensity"], c["concept_proximity"], marker="o", label=f"{pack} (ghost)")
    if placebo:
        c = ghost_curve(df, placebo, concept)
        ax.plot(c["intensity"], c["concept_proximity"], marker="x", linestyle="--",
                color="gray", label=f"{placebo} (null)")
    ax.set_xlabel("Dose (intensity)")
    ax.set_ylabel(f"Off-prompt proximity: '{concept}'")
    ax.set_title("Latent Specter: ghost prevalence vs dose")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def ghost_montage_from_images(images_by_seed: Dict[int, "object"], out_path: str,
                              max_seeds: int = 12) -> bool:
    """Montage of high-dose generations where the ghost is strongest (needs saved images)."""
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
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis("off")
    for ax, seed in zip(axs, seeds):
        ax.imshow(images_by_seed[seed])
    fig.suptitle("Exhumed latent specters (high-dose DMT)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


def analyze(csv_path: str, outdir: str, treatments: List[str], placebo: Optional[str],
            concept: str, fp_threshold: float = 0.25, plots: bool = True) -> Dict[str, object]:
    os.makedirs(outdir, exist_ok=True)
    df = _stats.load_long(csv_path)
    packs = set(df["pack"].unique())
    treatments = [t for t in treatments if t in packs]

    curves = []
    for t in treatments:
        c = ghost_curve(df, t, concept)
        c.insert(0, "pack", t)
        curves.append(c)
    if placebo in packs:
        c = ghost_curve(df, placebo, concept)
        c.insert(0, "pack", placebo)
        curves.append(c)
    if curves:
        pd.concat(curves).to_csv(os.path.join(outdir, "ghost_curves.csv"), index=False)

    contrasts = []
    if placebo in packs:
        for t in treatments:
            row = {"treatment": t, "placebo": placebo, "concept": concept}
            row.update(placebo_contrast(df, t, placebo, concept))
            contrasts.append(row)
    contrast_df = pd.DataFrame(contrasts)
    if not contrast_df.empty:
        contrast_df.to_csv(os.path.join(outdir, "placebo_contrast.csv"), index=False)

    null_packs = [p for p in ([placebo] if placebo in packs else []) if p]
    fp = pareidolia_fp_rate(df, null_packs, concept, fp_threshold) if null_packs else {}

    if plots and treatments:
        plot_ghost_prevalence(df, treatments, placebo if placebo in packs else None, concept,
                              os.path.join(outdir, "ghost_prevalence.png"))

    if not contrast_df.empty:
        print("Placebo contrast (treatment vs placebo, top dose):")
        print(contrast_df[["treatment", "dose", "mean_treatment", "mean_placebo",
                           "cohens_d", "cliffs_delta"]].to_string(index=False))
    if fp:
        print(f"\nPareidolia FP rate (null, concept>{fp['threshold']}): {fp.get('fp_rate')}")
    return {"contrast": contrast_df, "fp_rate": fp}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latent Specter: statistical ghost detection")
    ap.add_argument("--in", dest="csv_path", required=True)
    ap.add_argument("--outdir", default="outputs/dose_response/latent_specter")
    ap.add_argument("--treatments", default="dmt,lsd")
    ap.add_argument("--placebo", default="placebo")
    ap.add_argument("--concept", default="a human figure")
    ap.add_argument("--fp-threshold", type=float, default=0.25)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    treatments = [t.strip() for t in args.treatments.split(",") if t.strip()]
    analyze(args.csv_path, args.outdir, treatments, args.placebo, args.concept,
            fp_threshold=args.fp_threshold, plots=not args.no_plots)


if __name__ == "__main__":
    main()
