#!/usr/bin/env python3
"""
Decision matrix for the dose-response pilot (issue #11).

Scores the four "hero" threads on the criteria from the plan and ranks them so the headline
for the Unprompted.au talk is a data-driven pick rather than a hunch:

  * **statistical_strength** — the thread's headline effect (effect size / monotonicity),
    normalized across threads (data-derived).
  * **visual_drama** — how striking the sweep looks, from a data proxy (e.g. the dynamic
    range of the hero metric), normalized across threads.
  * **security_relevance** — fixed per-thread weight reflecting fit to a security audience.
  * **novelty** — fixed per-thread weight reflecting how new/surprising the result is.

Pure and unit-tested: the pilot driver (`scripts/run_pilot.py`) produces the per-thread
signal dicts from real (or dry-run) generations and passes them here.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Default criterion weights (sum to 1.0). Tunable by the caller.
DEFAULT_WEIGHTS = {
    "statistical_strength": 0.40,
    "visual_drama": 0.25,
    "security_relevance": 0.20,
    "novelty": 0.15,
}

# Fixed qualitative priors per thread (0..1), reflecting the plan's framing for a
# security-research audience. Data-derived criteria override the guesswork below.
THREAD_PRIORS = {
    "latent_specter":  {"security_relevance": 0.85, "novelty": 1.00},
    "safety_boundary": {"security_relevance": 1.00, "novelty": 0.80},
    "mode_collapse":   {"security_relevance": 0.70, "novelty": 0.60},
    "vitals_monitor":  {"security_relevance": 0.60, "novelty": 0.70},
}


def _minmax(values: List[float]) -> List[float]:
    """Min-max normalize to [0, 1]; all-equal -> all 0.5; NaNs -> 0."""
    arr = np.array([v if v is not None and np.isfinite(v) else np.nan for v in values], dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return [0.0] * len(values)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    out = []
    for v in arr:
        if not np.isfinite(v):
            out.append(0.0)
        elif hi > lo:
            out.append((v - lo) / (hi - lo))
        else:
            out.append(0.5)
    return out


def score_threads(thread_signals: List[Dict], weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Rank threads from their signal dicts.

    Each signal dict needs: ``thread`` (key into THREAD_PRIORS), ``stat_strength_raw``
    (data-derived, e.g. Cohen's d or |delta|), ``visual_drama_raw`` (data proxy). Optional
    ``security_relevance`` / ``novelty`` override the priors.

    Returns a DataFrame sorted by ``total`` descending, with normalized per-criterion columns.
    """
    weights = weights or DEFAULT_WEIGHTS
    if not thread_signals:
        return pd.DataFrame()

    stat_norm = _minmax([s.get("stat_strength_raw") for s in thread_signals])
    drama_norm = _minmax([s.get("visual_drama_raw") for s in thread_signals])

    rows = []
    for i, s in enumerate(thread_signals):
        thread = s["thread"]
        priors = THREAD_PRIORS.get(thread, {})
        sec = s.get("security_relevance", priors.get("security_relevance", 0.5))
        nov = s.get("novelty", priors.get("novelty", 0.5))
        crit = {
            "statistical_strength": stat_norm[i],
            "visual_drama": drama_norm[i],
            "security_relevance": float(sec),
            "novelty": float(nov),
        }
        total = sum(weights.get(k, 0.0) * v for k, v in crit.items())
        rows.append({"thread": thread, **crit, "total": total,
                     "stat_strength_raw": s.get("stat_strength_raw"),
                     "visual_drama_raw": s.get("visual_drama_raw")})
    df = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def recommend(thread_signals: List[Dict], weights: Optional[Dict[str, float]] = None) -> Dict:
    """Return the ranked table plus the headline pick and runners-up."""
    df = score_threads(thread_signals, weights)
    if df.empty:
        return {"table": df, "headline": None, "supporting": []}
    return {
        "table": df,
        "headline": df.iloc[0]["thread"],
        "supporting": list(df.iloc[1:3]["thread"]) if len(df) > 1 else [],
    }
