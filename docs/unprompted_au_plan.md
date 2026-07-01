# Dose-Response "Vitals Monitor" for Neuromodulated Diffusion — Unprompted.au Plan

> Planning + build tracker for upgrading the visual neuromodulation results from
> anecdotal "trippy trees" into a statistically rigorous **dose-response pharmacodynamics**
> study for submission to **Unprompted.au** (AI × cybersecurity research conference).

## Context

`neuromod-llm-poc` applies inference-time "neuromodulation packs" to LLMs and, cross-modally,
to Stable Diffusion / SDXL-Turbo (`demo/image_generation_demo.py`). The image results are
currently **anecdotal**: a one-shot grid of "trips" (`fig:visual_trips_complete`), a static
6-row spectral table (`tab:spectral_stats`, N=128, **no dose axis**), and a single "Latent
Ghost" DMT image (`fig:dmt_ghost`).

Unprompted.au rewards quantitative, honest, security-framed results. The paper's **own
Limitations section already concedes** the two gaps this audience will attack:

- *"Absence of Dose–Response and Monotonicity Curves"*
- *"Spectral Safety Auditing"* — listed only as **Future Work**.

**Goal:** build a measurement + statistics layer that turns the existing `intensity` knob into
rigorous dose-response curves (with CIs, monotonicity tests, breakpoints, effect sizes), run it
across four "hero" threads, and **down-select** to the most visually + intellectually stunning
headline. The dosing knob and pipeline already exist; the measurement layer did not — that is
the build.

## The four hero threads (run all at pilot scale, then down-select)

| Thread | Claim to prove | Key metric(s) | Security framing |
|---|---|---|---|
| **A. Latent Specter** | Alignment buries high-energy structural priors, exhumed under entropy | off-prompt CLIP concept proximity; latent structured energy; placebo control | mechanistic unlearning / red-team audit |
| **B. Architectural jailbreak** | High-intensity dosing flips visual safety filters on benign prompts | safety-trigger-rate vs dose; CLIP proximity to flagged concepts | entropy as an architectural jailbreak vector |
| **C. Cocaine Crunch** | Stimulants → spectral constriction → mode collapse / OOD failure | inter-seed diversity ↓; spatial variance/energy ↓; OOD-capacity | robustness / capability-suppression |
| **D. Vitals monitor** | Frame-by-frame disintegration of priors, synced to live metric graph | CLIP drop + spectral energy explosion vs fine dose grid | the live-demo centerpiece |

Down-selection: pilot all four (N=16 seeds, coarse grid) → score on statistical strength ×
visual drama × security relevance × novelty → run full **N=100** only on the winner(s).

## Dose-response design

- **Independent variable:** `intensity` swept `0.0 → 1.0` step `0.1` (11 points); dose 0.0 is the
  sober baseline for the *same* seed, making every metric a within-seed delta.
- **Sample size:** N=100 fixed seeds per (pack × dose), re-seeded before every generation.
- **Dependent variables (per generation):**
  - **CLIP semantic drift** — cosine(prompt, image); the "is it still a tree" vital.
  - **LPIPS** — vs baseline and vs previous dose (the break-point signal).
  - **SSIM / MS-SSIM** — structural integrity vs baseline.
  - **FFT scalars** — spectral energy, spatial variance, radial high/low band ratio, spectral
    entropy, on pixel + each latent channel (the scalar reducer behind Table 2).
  - **Inter-seed diversity** — mean pairwise LPIPS across seeds at a dose (mode-collapse metric).
- **Statistics:** per-dose bootstrap 95% CI ribbons; Spearman ρ + Mann-Kendall monotonicity;
  changepoint "cliff" detection; Benjamini-Hochberg FDR across all (pack, metric) tests.

## Status

### Phase 0 — shared measurement layer — **DELIVERED** (GPU-free, unit-tested)

| Component | File | Notes |
|---|---|---|
| Scalar metrics | `neuromod/metrics/pharmacodynamics.py` | CLIP/LPIPS (lazy, optional) + FFT scalars/SSIM/diversity (always) |
| `FrequencyAnalyzer` scalar refactor | `demo/image_generation_demo.py` | now returns the Table-2 scalars alongside its plot |
| Batch runner | `demo/dose_response_runner.py` | N-seed × dose sweep, tidy CSV, resumable, `--dry-run` |
| Stats + plots | `analysis/dose_response_stats.py` | bootstrap CIs, monotonicity, breakpoints, BH-FDR, ribbons |
| Unit tests | `tests/test_pharmacodynamics.py`, `tests/test_dose_response_stats.py` | 25 tests, synthetic tensors, no GPU |

Design note: learned metrics (CLIP, LPIPS) require torch + their backends and are imported
lazily — if unavailable, those columns are silently omitted so a run never fails for lack of a
GPU. Frequency-domain and structural metrics depend only on numpy/scikit-image.

### Phase 1 — pilot all four threads (N=16) on a GPU box → decision matrix

To build (thread-specific, sit on top of the Phase 0 layer):
- `analysis/latent_specter.py` — off-prompt concept proximity + placebo null + pareidolia FP rate.
- `analysis/safety_boundary.py` — re-enable SD `StableDiffusionSafetyChecker` **and** an
  independent CLIP-NSFW / violence oracle; benign prompts only; SFW-redact flagged outputs.
- `analysis/mode_collapse.py` — inter-seed diversity collapse + OOD-capacity curves.
- `demo/vitals_monitor.py` — fine-grid single-seed frames + synced vitals panel → mp4 + HTML slider.

### Phase 2 — full N=100 on the winning thread(s)

### Phase 3 — figures, vitals video + HTML slider, paper/talk update

Retire the paper's "Absence of Dose-Response" limitation; replace static Table 2 + ghost
anecdote with dose-response curves.

## How to run

```bash
# Validate the whole pipeline with no GPU / no model weights (synthetic generator):
python demo/dose_response_runner.py --dry-run --packs lsd,cocaine --seeds 8 \
    --out outputs/dose_response/pilot.csv
python analysis/dose_response_stats.py --in outputs/dose_response/pilot.csv \
    --outdir outputs/dose_response/analysis --plots

# Full study on a GPU box (SDXL-Turbo), N=100 seeds, 11 doses:
python demo/dose_response_runner.py --model sdxl-turbo --packs lsd,dmt,cocaine,amphetamine \
    --prompt "a tree" --seeds 100 --out outputs/dose_response/full.csv

# Unit tests (GPU-free):
python -m pytest tests/test_pharmacodynamics.py tests/test_dose_response_stats.py -q
```

## Ethics / safety guardrails (Thread B)

Benign prompts only; measure classifier activation / CLIP proximity, never attempt to *produce*
illegal or harmful imagery; SFW-redact any flagged output; keep intensity caps. An independent
oracle avoids circularity with the model's own safety checker. Either outcome — filters hold or
filters fail — is a strong, honest talk; the conference explicitly values honest failures.

## Risks

- **No GPU / no weights in the dev environment** → Phase 0 built + unit-tested on synthetic
  tensors; generation runs on the user's GPU (repo has `api/` + Vertex/CloudRun scaffolding).
- **Ghost pareidolia** → pre-registered template + placebo null + reported false-positive rate.
- **Safety-checker circularity** → independent oracle classifier.
- **Latent capture** → diffusers callback for pre-VAE latents (SDXL-Turbo = 4×64×64/128).
- **Compute** → staged pilot-then-full down-selection.
