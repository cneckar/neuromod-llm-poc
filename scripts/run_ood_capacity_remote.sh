#!/usr/bin/env bash
# "Cocaine Crunch" capacity-collapse study over the deployed RunPod worker (thread C+).
#
# Sweeps stimulant dose for an in-distribution prompt ("a tree") and a battery of compositional
# OOD prompts, then contrasts how fast CLIP prompt-adherence decays. If stimulants cause a genuine
# capacity failure (not just lower variance), adherence to the hard OOD prompts collapses faster and
# further than to the easy in-distribution one.
#
# Usage:
#   export RUNPOD_ENDPOINT_ID=xxx RUNPOD_API_KEY=yyy
#   scripts/run_ood_capacity_remote.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${RUNPOD_ENDPOINT_ID:?set RUNPOD_ENDPOINT_ID}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"

PACKS="${PACKS:-cocaine,amphetamine}"
SEEDS="${SEEDS:-100}"
STEP="${STEP:-0.1}"
MODEL="${MODEL:-sdxl-turbo}"
STEPS="${STEPS:-4}"
OUTDIR="${OUTDIR:-outputs/ood}"
mkdir -p "$OUTDIR"

# Pull the in-distribution + OOD prompt battery from the analysis module (single source of truth).
# (Captured into a TSV string rather than `readarray`, which macOS's stock bash 3.2 lacks.)
PROMPTS_TSV="$(python3 - <<'PY'
import analysis.ood_capacity as oc
print(f"tree\t{oc.INDIST_PROMPT}")
for label, prompt in oc.OOD_PROMPTS.items():
    print(f"{label}\t{prompt}")
PY
)"

sweep () {  # label  prompt  — generates to $OUTDIR/$label.csv with LIVE progress
  local label="$1" prompt="$2" csv="$OUTDIR/$label.csv"
  echo ">> [$label] \"$prompt\"  ->  $csv"
  # Run in the foreground WITHOUT capturing stdout, so the runner's per-cell progress
  # (\"[i/total] pack i=.. seed=..\") streams to the terminal. (An earlier version wrapped
  # this in $(...) for `tail -1`, which swallowed all progress and looked like a hang.)
  python3 demo/dose_response_runner.py \
    --remote --model "$MODEL" --steps "$STEPS" --no-latents \
    --packs "$PACKS" --seeds "$SEEDS" --intensity-step "$STEP" \
    --prompt "$prompt" --out "$csv"
}

# Scale heads-up: the first cell may take a minute or two while the worker cold-starts.
NPROMPTS=$(printf '%s\n' "$PROMPTS_TSV" | grep -c .)
NPACKS=$(awk -F, '{print NF}' <<<"$PACKS")
NDOSE=$(python3 -c "print(int(round(1.0/$STEP))+1)")
echo "Sweeping $NPROMPTS prompts x $NPACKS packs x $NDOSE doses x $SEEDS seeds"
echo "  (~$((NPROMPTS*NPACKS*NDOSE*SEEDS)) remote generations; resumable — re-run to continue if interrupted)"
echo

OOD_ARGS=()
INDIST_ARG=""
# Here-string keeps the loop in the current shell (so INDIST_ARG/OOD_ARGS persist) and
# works on bash 3.2. The CSV path is deterministic, so we build the pair from known vars
# instead of capturing sweep's stdout (which would hide progress). Skip blank lines.
while IFS=$'\t' read -r label prompt; do
  [ -n "$label" ] || continue
  sweep "$label" "$prompt"
  pair="$label=$OUTDIR/$label.csv"
  if [ "$label" = "tree" ]; then INDIST_ARG="$pair"; else OOD_ARGS+=("$pair"); fi
done <<< "$PROMPTS_TSV"

OOD_JOINED=$(IFS=,; echo "${OOD_ARGS[*]:-}")
echo "Analyzing OOD capacity gap -> $OUTDIR/analysis"
python3 analysis/ood_capacity.py --indist "$INDIST_ARG" --ood "$OOD_JOINED" \
  --packs "$PACKS" --outdir "$OUTDIR/analysis"
echo "Done. ood_capacity_gap.csv + ood_capacity.png in $OUTDIR/analysis"
