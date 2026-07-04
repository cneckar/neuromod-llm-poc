#!/usr/bin/env bash
# Full N=100 fine-grid dose-response run over a deployed RunPod worker (SDXL-Turbo).
#
# Generation runs on the worker's GPU (task="image"); the pharmacodynamic metrics
# (SSIM / LPIPS / inter-seed diversity / CLIP drift / spectral energy) are computed locally,
# so this box needs the metric deps (torch + open_clip + lpips + scikit-image) but NO SD weights.
# The run is resumable — re-run the same command after an interruption and it skips finished cells.
#
# Usage:
#   export RUNPOD_ENDPOINT_ID=xxxxx RUNPOD_API_KEY=yyyyy
#   scripts/run_dose_response_remote.sh
#
# Tunables (env):
#   PACKS     packs to sweep            (default: the pilot's down-selected winners + controls)
#   SEEDS     fixed seeds per cell      (default: 100)
#   STEP      dose grid step 0.0..1.0   (default: 0.05  -> 21 doses; use 0.1 for a faster 11-dose grid)
#   PROMPT    fixed text prompt         (default: "a tree")
#   CONCEPTS  off-prompt CLIP concepts  (default: "a human figure,a face"  -> the Latent-Specter probe)
#   MODEL     SD model / IMAGE_MODEL    (default: sdxl-turbo)
#   STEPS     diffusion steps per image (default: 4)
#   OUT       results CSV path          (default: outputs/dose_response/sdxl_turbo_n100.csv)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"

PACKS="${PACKS:-cocaine,amphetamine,dmt,lsd,placebo}"
SEEDS="${SEEDS:-100}"
STEP="${STEP:-0.05}"
PROMPT="${PROMPT:-a tree}"
CONCEPTS="${CONCEPTS:-a human figure,a face}"
MODEL="${MODEL:-sdxl-turbo}"
STEPS="${STEPS:-4}"
OUT="${OUT:-outputs/dose_response/sdxl_turbo_n100.csv}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$(dirname "$OUT")/analysis}"

# Rough size: (#packs) x (1/STEP + 1) doses x SEEDS images.
NPACKS=$(awk -F, '{print NF}' <<<"$PACKS")
NDOSE=$(python3 -c "print(int(round(1.0/$STEP))+1)")
TOTAL=$(python3 -c "print($NPACKS*$NDOSE*$SEEDS)")
echo "Endpoint : $RUNPOD_ENDPOINT_ID"
echo "Packs    : $PACKS"
echo "Grid     : 0.0..1.0 step $STEP  ($NDOSE doses)   Seeds: $SEEDS"
echo "Total    : ~$TOTAL image generations -> $OUT"
echo

# 0) Preflight: the driver box scores metrics locally, so it needs these — fail fast if missing
#    (otherwise cells "succeed" with SSIM/LPIPS/diversity silently omitted).
echo "[0/3] Checking local metric deps…"
python3 - <<'PY' || { echo "  -> install the missing deps above, then re-run."; exit 1; }
import importlib.util as u
need = {"torch": "torch", "PIL": "pillow", "numpy": "numpy",
        "skimage": "scikit-image", "lpips": "lpips", "open_clip": "open_clip_torch"}
missing = [pip for mod, pip in need.items() if u.find_spec(mod) is None]
if missing:
    print("  MISSING:", " ".join(missing))
    print("  pip install " + " ".join(missing))
    raise SystemExit(1)
print("  all metric backends present (torch, PIL, numpy, scikit-image, lpips, open_clip)")
PY

# 1) Warm the worker so the first image doesn't eat the whole cold start mid-loop.
echo "[1/3] Warming worker (loads the SD model once)…"
python3 - <<PY || echo "  (warmup skipped: $?)"
from api.runpod_client import RunPodModelInterface
import os
c = RunPodModelInterface(os.environ["RUNPOD_ENDPOINT_ID"], os.environ["RUNPOD_API_KEY"])
print("  ", c.generate_image("warmup ping", intensity=0.0, seed=0, steps=1, image_model="$MODEL").get("image", "")[:32], "…")
PY

# 2) The sweep (resumable). --remote reads RUNPOD_ENDPOINT_ID / RUNPOD_API_KEY.
echo "[2/3] Sweeping (resumable — safe to re-run if interrupted)…"
python3 demo/dose_response_runner.py \
  --remote --model "$MODEL" --steps "$STEPS" \
  --packs "$PACKS" --seeds "$SEEDS" --intensity-step "$STEP" \
  --prompt "$PROMPT" --concepts "$CONCEPTS" \
  --out "$OUT"

# 3) Stats + ribbon plots (Spearman / Mann-Kendall / BH-FDR / EC50-Hill / breakpoints).
echo "[3/3] Aggregating stats + plots -> $ANALYSIS_DIR"
python3 analysis/dose_response_stats.py --in "$OUT" --outdir "$ANALYSIS_DIR" --plots

echo
echo "Done. Curves + trend tables in: $ANALYSIS_DIR"
echo "  (dose curves CSV, trend_summary.csv, and per-metric ribbon plots)"
