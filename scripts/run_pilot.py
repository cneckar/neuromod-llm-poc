#!/usr/bin/env python3
"""
Dose-response pilot orchestrator (issue #11).

Runs all four "hero" threads at a reduced budget (default N=16 seeds, coarse dose grid),
extracts each thread's headline signal, and scores them with the decision matrix to
recommend the talk headline + supporting threads — so the "most stunning" pick is
data-driven. On a GPU this drives the real generations; ``--dry-run`` uses the synthetic
generator to validate the full pipeline without a GPU.

    # GPU-free plumbing check:
    python scripts/run_pilot.py --dry-run --seeds 4 --outdir outputs/pilot_demo

    # Real pilot on a GPU box:
    python scripts/run_pilot.py --model sdxl-turbo --seeds 16 --outdir outputs/pilot

Note: thread A needs off-prompt CLIP concepts and thread B needs the safety oracle wired
into generation (tracked in #21); in ``--dry-run`` those signals are absent, so those threads
score on visual-drama + priors only. Thread C and the visual-drama proxy run fully synthetic.
"""

import argparse
import importlib.util
import os
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("dose_response_runner", "demo/dose_response_runner.py")
stats = _load("dose_response_stats", "analysis/dose_response_stats.py")
mode_collapse = _load("mode_collapse", "analysis/mode_collapse.py")
latent_specter = _load("latent_specter", "analysis/latent_specter.py")
safety_boundary = _load("safety_boundary", "analysis/safety_boundary.py")
decision_matrix = _load("decision_matrix", "analysis/decision_matrix.py")

# Per-thread config: packs to sweep, the "hero" metric whose dynamic range proxies visual
# drama, and any extra generation flags.
THREADS = [
    {"key": "latent_specter",  "packs": ["dmt", "lsd", "placebo"], "hero": "latent_spectral_entropy",
     "concepts": ["a human figure", "a face"]},
    {"key": "safety_boundary", "packs": ["lsd", "dmt", "cocaine"], "hero": "pixel_high_low_ratio"},
    {"key": "mode_collapse",   "packs": ["cocaine", "amphetamine"], "hero": "latent_variance"},
    {"key": "vitals_monitor",  "packs": ["lsd"], "hero": "latent_energy"},
]


def _metric_range(df, packs: List[str], metric: str) -> float:
    """Dynamic range (max-min of per-dose mean) of a metric across the given packs."""
    sub = df[df["pack"].isin(packs)]
    if sub.empty:
        return np.nan
    curves = stats.dose_curves(sub, n_boot=500)
    m = curves[curves["metric"] == metric]
    if m.empty:
        return np.nan
    return float(np.nanmax(m["mean"]) - np.nanmin(m["mean"]))


def _thread_signal(cfg: Dict, df) -> Dict:
    """Extract a thread's statistical-strength + visual-drama signals from its CSV."""
    key = cfg["key"]
    drama = _metric_range(df, cfg["packs"], cfg["hero"])
    stat = np.nan

    if key == "mode_collapse":
        c = mode_collapse.classify_stimulant(df, "cocaine")
        stat = abs(c.get("d_diversity") if np.isfinite(c.get("d_diversity", np.nan))
                   else c.get("d_variance", np.nan))
    elif key == "latent_specter":
        con = latent_specter.placebo_contrast(df, "dmt", "placebo", cfg["concepts"][0])
        stat = abs(con.get("cohens_d", np.nan)) if np.isfinite(con.get("cohens_d", np.nan)) else np.nan
    elif key == "safety_boundary":
        summ = safety_boundary.summarize(df, cfg["packs"])
        stat = float(summ["delta"].abs().max()) if not summ.empty else np.nan
    elif key == "vitals_monitor":
        clip = _metric_range(df, cfg["packs"], "clip_prompt_similarity")
        stat = clip if np.isfinite(clip) else np.nan

    return {"thread": key, "stat_strength_raw": stat, "visual_drama_raw": drama}


def run_pilot(model: str, seeds: int, intensities: List[float], outdir: str,
              dry_run: bool, prompt: str = "a tree") -> Dict:
    os.makedirs(outdir, exist_ok=True)
    seed_list = list(range(seeds))
    signals = []

    # Load the model ONCE and reuse it across all threads (a fresh ModelGenerator per
    # thread would reload ~13GB of weights each time -- slow, and the silent reload gap
    # looks like a hang).
    generator = (runner.DryRunGenerator(prompt) if dry_run
                 else runner.ModelGenerator(model_name=model, prompt=prompt))

    for i, cfg in enumerate(THREADS, 1):
        print(f"[pilot] Thread {i}/{len(THREADS)}: {cfg['key']} "
              f"(packs={cfg['packs']}, seeds={len(seed_list)})", flush=True)
        csv_path = os.path.join(outdir, f"{cfg['key']}.csv")
        runner.run(
            generator=generator, packs=cfg["packs"], intensities=intensities, seeds=seed_list,
            prompt=prompt, csv_path=csv_path, concepts=cfg.get("concepts"),
            diversity_method="ssim" if dry_run else "auto", verbose=False,
        )
        df = stats.load_long(csv_path)
        signals.append(_thread_signal(cfg, df))

    rec = decision_matrix.recommend(signals)
    table = rec["table"]
    table.to_csv(os.path.join(outdir, "decision_matrix.csv"), index=False)

    print("\n=== Pilot decision matrix ===")
    print(table.to_string(index=False))
    print(f"\nHeadline pick: {rec['headline']}")
    print(f"Supporting:    {rec['supporting']}")
    if dry_run:
        print("\n[dry-run] latent_specter/safety signals are synthetic-absent; run on GPU "
              "with --concepts + safety oracle (#21) for the real ranking.")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dose-response pilot + decision matrix")
    ap.add_argument("--model", default="sdxl-turbo")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--intensities", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--outdir", default="outputs/pilot")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--prompt", default="a tree")
    args = ap.parse_args(argv)
    intensities = [float(x) for x in args.intensities.split(",") if x.strip() != ""]
    run_pilot(args.model, args.seeds, intensities, args.outdir, args.dry_run, args.prompt)


if __name__ == "__main__":
    main()
