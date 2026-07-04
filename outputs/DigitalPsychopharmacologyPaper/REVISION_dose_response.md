# Revision plan: dose–response pharmacodynamics (Unprompted.au reframe)

**Status:** draft against the *pilot* (SDXL-Turbo, N=16 seeds, 5-dose grid). Every number in
`dose_response_revision.draft.tex` is a **placeholder** marked `% TODO(N100)` — replace with the
N=100 / 21-dose (`step 0.05`) run from `scripts/run_dose_response_remote.sh` before submission.

## The reframe (why this lands with an AI-security audience)

Unprompted.au rewards *quantified, controlled, honestly-caveated security results* over "look how
trippy the model gets." The pilot lets us convert the paper's own conceded gaps into exactly that,
by re-casting three phenomena in security terms:

| Paper's current framing | Unprompted.au reframe |
|---|---|
| "The Shape of Madness" — packs produce vivid states | **A quantified control surface**: `intensity` is a monotone knob with a *sharp activation threshold* (breakpoint below the first dose step). A tiny, inference-time, seed-reproducible perturbation flips the output — the signature of a cheap, high-leverage control/attack primitive. |
| Table 2 — static N=128 point estimates of "Agitation vs Constriction" | **A robustness/availability failure**: under stimulant steering, inter-seed output **diversity collapses** monotonically (mode collapse). Cocaine collapses harder than amphetamine — the same split, now as continuous, effect-sized curves. |
| "Latent Ghost" — a single spooky DMT silhouette = "Latent Archaeology" | **A method with a measured false-positive rate.** With a **placebo control** (random-direction steering, matched magnitude), the off-prompt "human figure / face" signal rises under DMT/LSD *and* placebo — so the ghost is largely **pareidolia under any perturbation**, only weakly drug-specific. The *genuinely* drug-specific, placebo-null result is dose-dependent **semantic detachment** (CLIP prompt-adherence collapses under DMT, flat under placebo). |

The honest negative (the ghost is mostly an artifact) is not a retreat — for this crowd it is the
**headline methodological contribution**: a latent/spectral "safety audit" is only meaningful against
a placebo null and an FDR correction, or pareidolia manufactures "ghosts" of any concept you probe for.

## Per-edit summary (details + LaTeX in `dose_response_revision.draft.tex`)

1. **NEW Results subsection — "Dose–Response Pharmacodynamics of the Control Surface"** (insert after
   `tab:spectral_stats`, before the ghost subsection). Adds `fig:dose_response` (ribbon curves) +
   `tab:dose_response` (Spearman ρ, Mann–Kendall, BH-FDR q, EC50/Hill, breakpoint, Cohen's d).
   Establishes controllability (SSIM↓, LPIPS↑ sigmoid) and mode collapse (diversity↓, step-LPIPS↓).

2. **RETIRE the limitation** "Absence of Dose–Response and Monotonicity Curves" (subsection at l.888):
   replace with a one-paragraph pointer to the new section; the only residual caveat is pilot scale /
   single visual model, which the N=100 SDXL run closes.

3. **REWRITE the "Emergent Latent Specters" subsection** (l.830) as a placebo-controlled finding: the
   ghost is largely artifactual; the placebo-null result is DMT semantic detachment. Demote
   `fig:dmt_ghost` from "evidence" to "illustration," add a controlled 3-condition panel
   (`fig:specter_control`).

4. **STRENGTHEN "Spectral Safety Auditing" future work** (l.933) with the control requirement the
   pilot exposes: benign prompts, an *independent* oracle, a dose sweep, a placebo arm, and a reported
   false-positive rate. Note honestly that the safety-classifier arm is wired but not yet run
   (`analysis/safety_boundary.py` emitted only PD metrics in the pilot).

5. **ABSTRACT + CONCLUSION**: one clause each — foreground the control-surface / mode-collapse /
   placebo-controlled-audit framing; temper the "Latent Archaeology" claim with the placebo caveat.

## Reviewer-response experiments — now wired (code on `main`)

The conference reviewer demanded four things beyond the dose curves; all are now built and runnable
over the deployed RunPod worker (SDXL-Turbo), scored locally. Each fragment above is backed by a
script + analysis:

| Reviewer demand | Code | Run |
|---|---|---|
| **Quantify the Latent Specter statistically (~1000 gens), with a control** | `analysis/latent_specter.py` (placebo contrast + Cohen's d + pareidolia FP rate); `latent_*` now returned by the worker | main sweep (`--concepts`), then `latent_specter.py` |
| **Architectural jailbreak — do safety filters fail?** | `analysis/safety_boundary.py` (`SafetyOracle` CLIP-NSFW + `SDModelChecker`, two independent detectors, redaction) | `scripts/run_safety_boundary_remote.sh` |
| **Cocaine Crunch → can't render OOD** | `analysis/ood_capacity.py` (in-dist vs OOD adherence retention) | `scripts/run_ood_capacity_remote.sh` |
| **Visual pharmacodynamics video/slider synced to a live graph** | `demo/vitals_monitor.py --remote` (fine grid, dual-panel image+vitals, mp4/gif/HTML) | `python demo/vitals_monitor.py --remote --pack lsd --steps 0,0.02,…,1.0` |
| **Depressant (Morphine/Fentanyl) structural break point** | added to the main sweep default packs | `scripts/run_dose_response_remote.sh` |
| **Latent Spectral Energy tracking (their exact ask)** | `task="image"` now returns pre-VAE latents (`return_latents`) → full `latent_*` metrics over HTTP | `--remote` (latents on by default) |

Section 4.8 (Figure 8, the "vibes collage") is explicitly retired: rename to **Visual
Pharmacodynamics**, lead with `fig:dose_response` + a `vitals_monitor` still + a link/QR to the video.

## Placeholder → final checklist (run after N=100)

- [ ] Regenerate `fig:dose_response` from `analysis/dose_response_stats.py --plots`.
- [ ] Fill `tab:dose_response` from `mc_stats/trend_summary.csv` (ρ, MK z, `spearman_q`, `ec50`,
      `hill_slope`, `breakpoint_dose`, `cohens_d`).
- [ ] Report EC50 honestly: the pilot's fit (~0.07) is **below grid resolution**; the fine grid
      (step 0.05) resolves it — quote the fitted value only if the CI sits inside the grid.
- [ ] Replace `N=16` → `N=100`, "5-dose" → "21-dose", and every `% TODO(N100)` value.
- [ ] Add the placebo-control concept table (DMT / LSD / placebo × dose) to `fig:specter_control`.
