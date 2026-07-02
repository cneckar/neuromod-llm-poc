# Reproducibility guide

This project ships a single, tiered playbook that regenerates the key experiments and
supporting collateral behind the paper *Digital Psychopharmacology*
(`outputs/DigitalPsychopharmacologyPaper/Digital-Psychopharmacology.tex`) and writes a
**`REPRODUCTION_REPORT.md`** mapping every artifact to the figure/table/claim it supports.

Everything routes through **one** implementation, `scripts/reproduce.py`. There are three ways
to invoke it — all produce the same artifacts under `outputs/reproduction/`:

1. **Locally via the script:** `python scripts/reproduce.py --tier <0|1|2>`
2. **Locally via the notebook:** open [`notebooks/reproduce_paper_colab.ipynb`](notebooks/reproduce_paper_colab.ipynb)
   in Jupyter — it auto-detects that you're already in the repo and skips the clone/install cells.
3. **In Colab:** open the same notebook, pick a tier, *Run all*.

- **Determinism:** seed = 42 everywhere (`ReproducibilitySwitches`; `PYTHONHASHSEED=42`).
- **Provenance:** every run snapshots `outputs/reproduction/provenance/` — the git SHA, working-tree
  status, `pip freeze`, the resolved config, and `analysis/plan.yaml` — so a report is always
  traceable to an exact commit + dependency set.
- **Legacy command:** `python reproduce_results.py [--test-mode]` still works but now simply
  **forwards** to `scripts/reproduce.py` (`--test-mode` → `--tier 1`, default → `--tier 2`).

The orchestrator never re-implements an experiment — it shells out to the existing
`scripts/` / `analysis/` / `demo/` entry points with the right arguments, records
status/outputs/runtime per stage, and continues on failure so a partial environment still
produces a useful report.

## Compute tiers

| Tier | Needs | Text model | Reproduces | Time |
|---|---|---|---|---|
| **0** | CPU only | — | Validators, committed-data figures, and the whole visual dose-response pipeline in dry-run/synthetic form. Everything regenerable **without a GPU, model weights, or tokens**. | ~minutes |
| **1** | 1 GPU | `gpt2` (ungated) | Real SDXL-Turbo dose-response (Table 2 spectral, ghost, safety, collapse, vitals) + the text battery on gpt2 (endpoints → Table 1 stats, Figs 4 & 6). Paper **methodology + qualitative** claims on open weights. | ~30–60 min |
| **2** | 1 GPU + HF token | `meta-llama/Llama-3.1-8B-Instruct` (gated) | The text experiments on the paper's own model at paper scale (ablation, Lazarus, calibration, 13-pack panel). The only tier that reproduces the paper's **exact numbers**. | hours |

> **gpt2 vs Llama:** tiers 0–1 default to `gpt2` so anyone can run the full text pipeline with
> zero gated-model friction — the resulting numbers are *illustrative of the method*, not the
> paper's statistics. Only tier 2 on Llama-3.1-8B reproduces the reported effect sizes/p-values.

## Prerequisites

```bash
pip install -e .            # the neuromod package + core deps (torch, pandas, scipy, matplotlib…)
# Visual pipeline (tier 1+): diffusers + SDXL-Turbo + the lazy metric backends
pip install 'diffusers>=0.27' transformers accelerate open_clip_torch lpips scikit-image imageio seaborn
```

- **Tier 0** needs the base install (torch etc.). Stages whose optional deps are missing are
  marked `🔌 skipped-deps` in the report, not failed.
- **Tier 2** needs an **HF token with Llama-3.1 access**: accept the license at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>, then
  `export HUGGINGFACE_TOKEN=hf_...` (or `huggingface-cli login`).

## Running it

```bash
# Inspect the plan for any tier (runs nothing):
python scripts/reproduce.py --tier 2 --list

# Tier 0 — CPU, regenerate everything possible from committed data + dry-run:
python scripts/reproduce.py --tier 0

# Tier 1 — one GPU, real SDXL + gpt2 text (headline collateral):
python scripts/reproduce.py --tier 1 --seeds 16

# Tier 2 — faithful, paper-scale, gated model:
HUGGINGFACE_TOKEN=hf_... python scripts/reproduce.py --tier 2 \
    --model meta-llama/Llama-3.1-8B-Instruct --seeds 100

# Subset / resume:
python scripts/reproduce.py --tier 1 --only visual_pilot,dose_stats,latent_specter
python scripts/reproduce.py --tier 1 --skip endpoints    # (endpoints are the slow text stage)
```

Everything is written under `--outdir` (default `outputs/reproduction/`):
`REPRODUCTION_REPORT.md`, `manifest.json`, `figures/`, `pilot/`, `dose_stats/`, `endpoints/`, …
The runner is **resumable** — the visual runner skips already-computed rows, and
`calculate_endpoints.py` runs with `--skip-completed`.

## What each stage supports (artifact → paper claim)

| Stage | Tier | Supports |
|---|---|---|
| `blinding_audit` | 0 | §4.3 Double-Blindfold Protocol (no pack-name leakage) |
| `design_validation` | 0 | §4.4 Experimental Design (Latin square, blind codes) |
| `stats_validation` | 0 | §4.7 Statistical Analysis (FDR, mixed-effects, power) |
| `power_analysis` | 0 | §4.7 Power Analysis (N=126 for d=0.25) |
| `ablation_committed` | 0 | Figs *Stimulant Ceiling* / *LSD Rigidity* / *Ablation* (committed data; T2 regenerates) |
| `steering_committed` | 0 | §3 Steering Vector Construction (committed `.pt`; T1 regenerates) |
| `figure4_committed` | 0 | Fig *Cognitive Impact* (`fig:cognitive`; committed render) |
| `figure_emotion` | 0 | Fig *Emotional Signatures* (`fig:emotion`) — from committed emotion runs |
| `visual_pilot` | 0→1 | Fig *Visual Trips* / **Table 2** spectral / decision matrix + hero montage |
| `dose_stats` | 0→1 | **§6.2 Dose-Response & Monotonicity** — the paper's conceded gap, now curves + EC50 |
| `mode_collapse` | 0→1 | **Table 2** — cocaine constriction vs amphetamine agitation, as continuous curves |
| `latent_specter` | 1 | Fig *DMT Ghost* (`fig:dmt_ghost`) → statistical ghost prevalence vs placebo |
| `safety_boundary` | 1 | **§7.4 Spectral Safety Auditing** (Future Work → result): trigger-rate vs dose |
| `vitals` | 0→1 | Demo vehicle: dose slider (GIF + interactive HTML) |
| `steering_vectors` | 1 | §3 Steering Vector Construction (regenerated) |
| `endpoints` | 1 | **Table 1** primary-endpoint detection battery |
| `analyze_endpoints` | 1 | Table 1 statistics (paired tests, BH-FDR, effect sizes) |
| `figure_detection` | 1 | Fig *Detection Sensitivity* (`fig:sensitivity`) |
| `figure_radar` | 1 | Fig *Behavioral Radar* (`fig:radar`) |
| `export_ndjson` | 1 | §4.7 power-analysis input |
| `lsd_ablation` | 2 | Figs *Ablation* / *Stimulant Ceiling* / *LSD Rigidity* (paper-scale, on Llama) |
| `lazarus` | 2 | §7.3 Digital IV — bidirectional Morphine→Cocaine steering |
| `calibration` | 2 | §4 Stimulant Ceiling — entropy↓ but calibration-error↑ (ECE/MCE/Brier) |

## Running against a RunPod Serverless endpoint (scale-to-zero)

If you've deployed the endpoint (`deploy/runpod/`), you can run the study over HTTP and pay for
GPU-seconds only while the worker executes — no always-on box. The handler routes on an
`input.task` field so heavy work runs **on the worker** (which has the model in-process) and
artifacts land on the network volume. Drive it with the torch-free client:

```bash
export RUNPOD_ENDPOINT_ID=... RUNPOD_API_KEY=...
# 1) one-time: regenerate steering vectors for the served model (else pack steering is a no-op)
python scripts/run_remote_study.py --mode steering  --model openai/gpt-oss-120b
# 2) the full internal-telemetry battery (Table 1) per pack, run on the worker, JSON returned
python scripts/run_remote_study.py --mode endpoints --model openai/gpt-oss-120b --packs lsd,cocaine,morphine
python scripts/analyze_endpoints.py --input-dir outputs/remote_study/endpoints
# or a text-only behavioral dose sweep -> feeds the dose-response stats
python scripts/run_remote_study.py --mode behavioral --model openai/gpt-oss-120b --packs lsd,cocaine
python analysis/dose_response_stats.py --in outputs/remote_study/behavioral.csv --plots
```

Note: **gpt-oss-120b ≠ the paper's Llama-3.1-8B**, so this reproduces the *methodology*, not the
paper's exact numbers. See `deploy/runpod/README.md` for the H200/MXFP4 + network-volume setup.

## Notes & known caveats

- **Two visual findings upgrade paper gaps to results.** `dose_stats` addresses the paper's own
  §6.2 *"Absence of Dose–Response and Monotonicity Curves"* limitation, and `safety_boundary`
  operationalizes §7.4 *Spectral Safety Auditing* (benign prompts only; flagged output is
  redacted, never persisted as pixels).
- **Paper figure references are aligned with the generators.** The `.tex` now references the
  canonical `figure_N_*.png` names the generators emit (`figure_3_radar_plots.png`,
  `figure_5_emotion_signatures.png`), and a `\graphicspath` resolves both the committed figures
  (`outputs/`) and freshly reproduced ones (`outputs/reproduction/figures/`) — so a reproduction
  run drops straight into the paper build. (`figure_4_cognitive_impact.png` still has no
  standalone generator; it is surfaced from the committed render by the `figure4_committed` stage.)
- **Committed inputs** (so downstream stages run without re-generating): `outputs/ablation_experiments/`,
  `outputs/reports/emotion/`, `outputs/validation/`, `outputs/steering_vectors/`, `analysis/plan.yaml`,
  `datasets/steering_prompts.jsonl`. The `outputs/endpoints/` directory is **not** committed — the
  `endpoints` stage generates it (tier 1+).
