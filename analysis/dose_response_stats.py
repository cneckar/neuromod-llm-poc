#!/usr/bin/env python3
"""
Dose-response statistics and visualization for the neuromodulation pharmacodynamics study.

Consumes the tidy long-format CSV produced by ``demo/dose_response_runner.py``
(columns: pack, intensity, seed, metric, value) and produces the "vitals monitor"
evidence a security-research audience expects instead of eyeballed grids:

  * Per-dose mean with **95% bootstrap confidence intervals** (ribbon plots).
  * **Monotonicity** tests: Spearman rho (with p) and the non-parametric Mann-Kendall
    trend test -- does the metric move systematically with dose?
  * **Breakpoint** ("the cliff") detection: the dose at which the biggest step change
    occurs, via a simple piecewise-constant changepoint scan on the dose-mean curve.
  * **Benjamini-Hochberg FDR** correction across all (pack, metric) monotonicity tests.

All statistics are emitted as a summary CSV; ribbon plots are written per (pack, metric)
when matplotlib is available. Duplicate (pack, intensity, seed, metric) rows created by a
resumed run are de-duplicated (last value wins) before analysis.

Usage
-----
    python analysis/dose_response_stats.py --in outputs/dose_response/full.csv \
        --outdir outputs/dose_response/analysis --plots
"""

import argparse
import importlib.util
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def _load_by_path(module_name: str, rel_path: str):
    """Load a repo module directly by file path.

    Avoids triggering the heavy ``neuromod`` package ``__init__`` (which imports torch)
    and lets us reuse existing analysis code. Returns None if the module or its deps
    are unavailable, so the dose-response stats degrade gracefully in a minimal env.
    """
    try:
        path = os.path.join(_REPO_ROOT, rel_path)
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _ablations_fitter():
    """Reuse ``AblationsAnalyzer``'s EC50/Hill curve fitter (numpy/scipy math only)."""
    mod = _load_by_path("ablations_analysis", os.path.join("neuromod", "testing", "ablations_analysis.py"))
    return getattr(mod, "AblationsAnalyzer", None) if mod else None


def _cliffs_delta_fn():
    """Reuse ``StatisticalAnalyzer._calculate_cliffs_delta`` (self-contained rank math)."""
    mod = _load_by_path("statistical_analysis", os.path.join("analysis", "statistical_analysis.py"))
    cls = getattr(mod, "StatisticalAnalyzer", None) if mod else None
    if cls is None:
        return None
    # The method only touches its args, so bind with a throwaway receiver.
    return lambda control, treatment: cls._calculate_cliffs_delta(None, control, treatment)


# --------------------------------------------------------------------------------------
# Loading / cleaning
# --------------------------------------------------------------------------------------


def load_long(csv_path: str) -> pd.DataFrame:
    """Load and clean the long-format results CSV.

    * Coerces types, drops rows that cannot be parsed.
    * De-duplicates (pack, intensity, seed, metric) keeping the last write, so a
      resumed run's re-appended diversity rows do not double-count.
    """
    df = pd.read_csv(csv_path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna(subset=["value", "intensity", "metric", "pack"])
    df = df.drop_duplicates(subset=["pack", "intensity", "seed", "metric"], keep="last")
    return df


def _seed_level(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed metrics only (exclude aggregate 'ALL' rows like inter-seed diversity)."""
    return df[df["seed"].astype(str) != "ALL"]


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, n_boot: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Mean and percentile bootstrap CI for a 1D sample."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0}
    if values.size == 1:
        v = float(values[0])
        return {"mean": v, "ci_low": v, "ci_high": v, "n": 1}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, values.size, size=(n_boot, values.size))
    boot_means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.percentile(boot_means, 100 * alpha / 2)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
        "n": int(values.size),
    }


def mann_kendall(x: np.ndarray) -> Dict[str, float]:
    """Non-parametric Mann-Kendall trend test on an ordered sequence of dose-means.

    Returns S, the normalized z, a two-sided p-value, and the trend direction.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return {"S": np.nan, "z": np.nan, "p": np.nan, "trend": "insufficient"}
    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k + 1:] - x[k]))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    trend = "increasing" if z > 0 else ("decreasing" if z < 0 else "none")
    return {"S": float(s), "z": float(z), "p": float(p), "trend": trend}


def detect_breakpoint(doses: np.ndarray, means: np.ndarray) -> Dict[str, float]:
    """Find the dose with the largest single-step change ("the cliff").

    Reports the interval midpoint, the step magnitude, and the fraction of the total
    dynamic range concentrated in that one step (a crude sharpness score).
    """
    doses = np.asarray(doses, dtype=float)
    means = np.asarray(means, dtype=float)
    order = np.argsort(doses)
    doses, means = doses[order], means[order]
    if means.size < 2:
        return {"breakpoint_dose": np.nan, "step": np.nan, "sharpness": np.nan}
    steps = np.abs(np.diff(means))
    i = int(np.argmax(steps))
    total_range = float(np.nanmax(means) - np.nanmin(means)) or 1.0
    return {
        "breakpoint_dose": float((doses[i] + doses[i + 1]) / 2.0),
        "step": float(steps[i]),
        "sharpness": float(steps[i] / total_range),
    }


def fit_dose_response_curve(doses: np.ndarray, means: np.ndarray) -> Dict[str, Optional[float]]:
    """Fit a dose-response curve and extract EC50 / Hill slope (paper Figure 6).

    Reuses ``AblationsAnalyzer._fit_dose_response_curve`` +
    ``_calculate_ec50_and_hill_slope`` (which try sigmoid/linear/exponential/polynomial and
    pick the best R^2) rather than reinventing the fitter. Returns NaN/None fields if the
    ablations module (or its deps) is unavailable.
    """
    empty = {"ec50": np.nan, "hill_slope": np.nan, "curve_r2": np.nan, "curve_type": None}
    analyzer = _ablations_fitter()
    if analyzer is None:
        return empty
    doses = np.asarray(doses, dtype=float)
    means = np.asarray(means, dtype=float)
    if np.unique(doses).size < 3 or np.unique(np.round(means, 12)).size < 2:
        return empty
    try:
        # These methods only read their arguments, so a None receiver is safe.
        params, r2, ctype = analyzer._fit_dose_response_curve(None, list(doses), list(means))
        ec50, hill = analyzer._calculate_ec50_and_hill_slope(None, list(doses), list(means), params, ctype)
        return {
            "ec50": float(ec50) if ec50 is not None else np.nan,
            "hill_slope": float(hill) if hill is not None else np.nan,
            "curve_r2": float(r2),
            "curve_type": ctype,
        }
    except Exception:
        return empty


def cohens_d_paired(baseline: np.ndarray, treatment: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / std(diff, ddof=1).

    Mirrors the paired-d computation in ``statistical_analysis.paired_t_test`` so the
    dose-response effect sizes stay consistent with the rest of the study.
    """
    baseline = np.asarray(baseline, dtype=float)
    treatment = np.asarray(treatment, dtype=float)
    n = min(baseline.size, treatment.size)
    if n < 2:
        return np.nan
    diff = treatment[:n] - baseline[:n]
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0 if np.mean(diff) == 0 else np.nan
    return float(np.mean(diff) / sd)


def effect_size_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per (pack, metric) max effect size of any dose vs the dose-0 baseline.

    Computes paired Cohen's d and (reused) Cliff's delta for every dose>0 against dose 0,
    using the per-seed values, and keeps the dose with the largest |Cohen's d|.
    """
    seed_df = _seed_level(df)
    cliffs = _cliffs_delta_fn()
    rows = []
    for (pack, metric), grp in seed_df.groupby(["pack", "metric"]):
        base = grp[grp["intensity"] == 0.0].sort_values("seed")["value"].values
        if base.size < 2:
            continue
        best = {"cohens_d": np.nan, "cliffs_delta": np.nan, "dose_of_max": np.nan}
        best_abs = -1.0
        for intensity, sub in grp.groupby("intensity"):
            if float(intensity) == 0.0:
                continue
            treat = sub.sort_values("seed")["value"].values
            d = cohens_d_paired(base, treat)
            if np.isnan(d):
                continue
            if abs(d) > best_abs:
                best_abs = abs(d)
                delta = float(cliffs(base, treat)) if cliffs is not None else np.nan
                best = {"cohens_d": d, "cliffs_delta": delta, "dose_of_max": float(intensity)}
        if best_abs >= 0:
            rows.append({"pack": pack, "metric": metric, **best})
    return pd.DataFrame(rows)


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH-FDR adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


# --------------------------------------------------------------------------------------
# Aggregation pipeline
# --------------------------------------------------------------------------------------


def dose_curves(df: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    """Per (pack, metric, intensity) mean + bootstrap CI table (the ribbon data)."""
    seed_df = _seed_level(df)
    rows = []
    for (pack, metric), grp in seed_df.groupby(["pack", "metric"]):
        for intensity, sub in grp.groupby("intensity"):
            ci = bootstrap_ci(sub["value"].values, n_boot=n_boot)
            rows.append({"pack": pack, "metric": metric, "intensity": float(intensity), **ci})
    return pd.DataFrame(rows).sort_values(["pack", "metric", "intensity"]).reset_index(drop=True)


def trend_summary(curves: pd.DataFrame) -> pd.DataFrame:
    """Monotonicity + breakpoint summary per (pack, metric), with BH-FDR across all tests."""
    rows = []
    for (pack, metric), grp in curves.groupby(["pack", "metric"]):
        grp = grp.sort_values("intensity")
        doses = grp["intensity"].values
        means = grp["mean"].values
        if np.unique(doses).size < 3:
            continue
        # A metric that is flat across dose has no defined rank correlation; report null.
        if np.unique(np.round(means, 12)).size < 2:
            rho, rho_p = 0.0, 1.0
        else:
            rho, rho_p = stats.spearmanr(doses, means)
        mk = mann_kendall(means)
        bp = detect_breakpoint(doses, means)
        curve = fit_dose_response_curve(doses, means)
        rows.append({
            "pack": pack,
            "metric": metric,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "mk_z": mk["z"],
            "mk_p": mk["p"],
            "trend": mk["trend"],
            "breakpoint_dose": bp["breakpoint_dose"],
            "step": bp["step"],
            "sharpness": bp["sharpness"],
            "range": float(np.nanmax(means) - np.nanmin(means)),
            "ec50": curve["ec50"],
            "hill_slope": curve["hill_slope"],
            "curve_r2": curve["curve_r2"],
            "curve_type": curve["curve_type"],
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["spearman_q"] = benjamini_hochberg(out["spearman_p"].fillna(1.0).tolist())
        out["mk_q"] = benjamini_hochberg(out["mk_p"].fillna(1.0).tolist())
        out = out.sort_values(["pack", "sharpness"], ascending=[True, False]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------------------
# Plotting (optional)
# --------------------------------------------------------------------------------------


def plot_curves(curves: pd.DataFrame, outdir: str) -> int:
    """Write one ribbon plot per (pack, metric). Returns number of plots written."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[plots] matplotlib unavailable, skipping plots: {e}")
        return 0

    os.makedirs(outdir, exist_ok=True)
    written = 0
    for (pack, metric), grp in curves.groupby(["pack", "metric"]):
        grp = grp.sort_values("intensity")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grp["intensity"], grp["mean"], marker="o", color="#c0392b", label="mean")
        ax.fill_between(grp["intensity"], grp["ci_low"], grp["ci_high"],
                        alpha=0.25, color="#c0392b", label="95% CI")
        ax.set_xlabel("Dose (intensity)")
        ax.set_ylabel(metric)
        ax.set_title(f"{pack}: {metric} vs dose")
        ax.legend(fontsize=8)
        fig.tight_layout()
        safe = f"{pack}__{metric}".replace("/", "_").replace("::", "_")
        fig.savefig(os.path.join(outdir, f"{safe}.png"), dpi=140)
        plt.close(fig)
        written += 1
    return written


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def analyze(csv_path: str, outdir: str, plots: bool = False, n_boot: int = 10000) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    df = load_long(csv_path)
    curves = dose_curves(df, n_boot=n_boot)
    trends = trend_summary(curves)
    effects = effect_size_summary(df)
    if not trends.empty and not effects.empty:
        trends = trends.merge(effects, on=["pack", "metric"], how="left")

    curves_path = os.path.join(outdir, "dose_curves.csv")
    trends_path = os.path.join(outdir, "trend_summary.csv")
    curves.to_csv(curves_path, index=False)
    trends.to_csv(trends_path, index=False)

    n_plots = plot_curves(curves, os.path.join(outdir, "plots")) if plots else 0

    print(f"Loaded {len(df)} rows -> {curves['metric'].nunique()} metrics across "
          f"{curves['pack'].nunique()} packs.")
    print(f"Wrote curve table:   {curves_path} ({len(curves)} rows)")
    print(f"Wrote trend summary: {trends_path} ({len(trends)} tests)")
    if plots:
        print(f"Wrote {n_plots} ribbon plots to {os.path.join(outdir, 'plots')}")
    if not trends.empty:
        print("\nTop dose-response signals (by breakpoint sharpness):")
        cols = ["pack", "metric", "trend", "spearman_rho", "spearman_q", "breakpoint_dose", "sharpness"]
        print(trends[cols].head(10).to_string(index=False))
    return {"curves": curves_path, "trends": trends_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dose-response statistics for neuromodulation study")
    ap.add_argument("--in", dest="csv_path", required=True, help="Long-format results CSV")
    ap.add_argument("--outdir", default="outputs/dose_response/analysis", help="Output directory")
    ap.add_argument("--plots", action="store_true", help="Write ribbon plots (needs matplotlib)")
    ap.add_argument("--n-boot", type=int, default=10000, help="Bootstrap iterations")
    args = ap.parse_args(argv)
    analyze(args.csv_path, args.outdir, plots=args.plots, n_boot=args.n_boot)


if __name__ == "__main__":
    main()
