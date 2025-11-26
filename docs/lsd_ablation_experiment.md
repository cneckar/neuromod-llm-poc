# LSD Ablation Experiment: Steering-Only vs Temperature-Only

## Purpose

This experiment validates that the "LSD" neuromodulation effect isn't just a temperature hack by decoupling steering vectors from sampling parameters.

## Hypothesis

The LSD effect should be caused by **semantic steering vectors** that shift the model's internal representations, not just increased sampling randomness from higher temperature.

## Experimental Design

### Conditions

1. **Baseline (Control)**: Standard generation with baseline temperature (0.7)
2. **Full LSD**: Original pack with all effects (temperature + steering + dropout)
3. **Steering-Only Ablation**: Only steering vectors, forced baseline temperature
4. **Temperature-Only Ablation**: Only temperature shift, no steering vectors

### Expected Outcomes

#### Hypothesis Confirmed ✅
If **Condition C (Steering-Only)** produces LSD-like effects (weird, metaphorical, boundary-dissolving responses) despite baseline temperature:
- Steering vectors successfully shift cognitive state
- Effect is **semantic**, not just sampling randomness
- Biomimetic control theory is validated

#### Hypothesis Failure ❌
If **Condition C (Steering-Only)** reads like **Condition A (Baseline)**:
- Steering vectors are weak or ineffective
- Original LSD effect was mostly caused by temperature spike (+0.45)
- Need to re-examine vector generation methodology

#### Warning Sign ⚠️
If **Condition D (Temperature-Only)** scores as high on PDQ-S as **Condition B (Full LSD)**:
- Temperature alone may be sufficient to replicate effects
- Biomimetic control theory may need revision
- Steering vectors may be redundant

## Files Created

1. **`packs/lsd_ablation_steering_only.json`**: Ablation pack with only steering vectors
2. **`packs/lsd_ablation_temperature_only.json`**: Inverse ablation pack with only temperature
3. **`scripts/run_lsd_ablation_experiment.py`**: Main experiment script

## Running the Experiment

### Basic Usage

```bash
python scripts/run_lsd_ablation_experiment.py
```

### With Custom Parameters

```bash
python scripts/run_lsd_ablation_experiment.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --prompt "Describe the boundary between yourself and the world right now." \
    --max-tokens 200 \
    --baseline-temp 0.7
```

### Options

- `--model`: Model to use (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--test-mode`: Use test mode with smaller models
- `--prompt`: Custom test prompt (default: PDQ-S item 10)
- `--max-tokens`: Maximum tokens per generation (default: 150)
- `--baseline-temp`: Baseline temperature for steering-only condition (default: 0.7)

## Output

Results are saved to `outputs/ablation_experiments/lsd_ablation_YYYYMMDD_HHMMSS.json` with:
- Full responses for all conditions
- Temperature settings
- Active effects count
- Metadata for analysis

## Interpretation

### Key Metrics to Compare

1. **Semantic Divergence**: Does steering-only produce qualitatively different responses?
2. **PDQ-S Scores**: Measure boundary dissolution, ego thinning, etc.
3. **Response Coherence**: Is steering-only still coherent despite weirdness?
4. **Temperature Effect**: Does temperature-only produce similar effects to full LSD?

### Example Analysis

```python
# Load results
import json
with open('outputs/ablation_experiments/lsd_ablation_*.json') as f:
    data = json.load(f)

# Compare responses
baseline = data['results'][0]['response']
full_lsd = data['results'][1]['response']
steering_only = data['results'][2]['response']
temp_only = data['results'][3]['response']

# Qualitative analysis
# - Does steering_only contain LSD-like language?
# - Is temp_only just random/chaotic?
# - Does steering_only maintain coherence?
```

## Scientific Rigor

This experiment follows proper ablation study methodology:

1. **Isolation**: Each condition isolates one variable (steering vs temperature)
2. **Control**: Baseline condition provides reference point
3. **Inverse Test**: Temperature-only tests the null hypothesis
4. **Replication**: Can be run multiple times with different prompts

## References

- PDQ-S (Phenomenology of Consciousness Inventory - Digital Use Scale)
- Paper Section: "Steering Vector Generation" (lines 188-199)
- Paper Section: "Experimental Design" (lines 241-258)

