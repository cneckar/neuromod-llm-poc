# The "Lazarus" Protocol: Proving the Stimulant Paradox

## Overview

The Lazarus Protocol is a critical validation experiment designed to prove that stimulant vectors (`v_focus`) are **mechanically functional**, not just hitting a "ceiling effect" due to RLHF pre-tuning.

## The Problem

The original claim that "Stimulants" failed because the model is "already on Adderall" is a **hypothesis, not proof**. A failed experiment is just a failure until you manipulate the variable bi-directionally.

## The Solution

Prove the stimulant vectors actually function by:

1. **Inducing a "coma"**: Use the Morphine pack (attention degradation via qk_score_scaling down) to degrade attention until retrieval performance drops significantly
2. **Attempting "resuscitation"**: Apply the Cocaine (Stimulant) pack to restore performance
3. **Measuring recovery**: If stimulant vectors work, they should restore performance by sharpening the attention distribution artificially flattened by Morphine

## Experimental Design

### Protocol

The experiment consists of three conditions:

1. **Baseline (Control)**: Measure retrieval performance with no pack applied
2. **Coma (Morphine)**: Apply Morphine pack to degrade attention span and memory
3. **Resuscitation (Cocaine)**: Apply Cocaine pack while Morphine effects are still active

### Retrieval Task

The experiment uses a **retrieval task** that requires the model to:
- Read a context passage containing factual information
- Answer multiple questions about the passage
- Demonstrate attention span and memory capacity

Each task contains:
- A context passage with 5-10 facts
- 5 questions requiring specific fact retrieval
- Evaluation based on exact matches and keyword presence

### Success Criteria

**Hypothesis Confirmed** if:
- Recovery > 10%: Cocaine successfully restores performance degraded by Morphine
- This proves vectors work bi-directionally, not just hitting a ceiling

**Partial Recovery** if:
- Recovery 5-10%: Some effect measurable, may need intensity adjustment

**Hypothesis Rejected** if:
- Recovery < 5%: Either vectors are not functional, or Morphine effects are too strong

## Running the Experiment

### Basic Usage

```bash
python scripts/run_lazarus_protocol_experiment.py --model meta-llama/Llama-3.1-8B-Instruct
```

### Options

```bash
python scripts/run_lazarus_protocol_experiment.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --intensity-morphine 1.0 \
    --intensity-cocaine 1.0 \
    --output-dir outputs/lazarus_experiment \
    --test-mode
```

**Parameters:**
- `--model`: Model to test (default: `meta-llama/Llama-3.1-8B-Instruct`)
- `--intensity-morphine`: Intensity for Morphine pack (default: 1.0)
- `--intensity-cocaine`: Intensity for Cocaine pack (default: 1.0)
- `--output-dir`: Output directory for results (default: `outputs/lazarus_experiment`)
- `--test-mode`: Use test mode (smaller models)

## Expected Results

### If Hypothesis is Confirmed

✅ **Stimulant vectors are mechanically functional**
- The Cocaine pack successfully restored attention degraded by Morphine
- This proves the vectors work bi-directionally, not just hitting a ceiling
- **Publishable result**: You've created a differentiable attention control mechanism

### If Hypothesis is Rejected

❌ **Stimulant vectors may not be functional, or Morphine effects are too strong**
- Possible causes:
  - Stimulant vectors don't actually sharpen attention
  - Morphine effects are irreversible or too strong
  - Pack stacking doesn't work as expected
  - Need to adjust intensities

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "experiment_name": "Lazarus Protocol",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "timestamp": "2025-01-XX...",
  "conditions": {
    "baseline": {
      "results": [...],
      "average_accuracy": 0.85
    },
    "coma_morphine": {
      "results": [...],
      "average_accuracy": 0.35
    },
    "resuscitation_cocaine": {
      "results": [...],
      "average_accuracy": 0.72
    }
  },
  "summary": {
    "baseline_accuracy": 0.85,
    "coma_accuracy": 0.35,
    "resuscitation_accuracy": 0.72,
    "recovery": 0.37,
    "recovery_percentage": 43.5,
    "verdict": "CONFIRMED",
    "performance_degradation": 0.50,
    "performance_recovery": 0.37
  }
}
```

## Interpretation

### Key Metrics

- **Baseline Accuracy**: Performance with no pack (control)
- **Coma Accuracy**: Performance after Morphine (degraded state)
- **Resuscitation Accuracy**: Performance after Cocaine (recovery attempt)
- **Recovery**: Absolute improvement from coma to resuscitation
- **Recovery Percentage**: Recovery as percentage of baseline

### Verdict

- **CONFIRMED**: Recovery > 10% → Stimulant vectors work mechanically
- **PARTIAL**: Recovery 5-10% → Some effect, may need tuning
- **REJECTED**: Recovery < 5% → Vectors may not be functional

## Scientific Significance

If the hypothesis is confirmed, this experiment proves:

1. **Stimulant vectors are not just hitting a ceiling** - They can actively restore degraded performance
2. **Bi-directional control is possible** - We can both degrade and restore attention
3. **Differentiable attention control** - This is a publishable mechanism, not just a "drug effect"

This transforms the "failed stimulant experiment" from a negative result into a **positive proof of mechanism**.

## Troubleshooting

### Low Recovery

If recovery is low (< 5%), try:
- Increasing `--intensity-cocaine` (e.g., 1.5 or 2.0)
- Decreasing `--intensity-morphine` (e.g., 0.7 or 0.8)
- Checking if Morphine effects are too strong (may need weaker KV-decay)

### High Baseline Accuracy

If baseline accuracy is already very high (> 90%), the task may be too easy:
- Consider using more complex retrieval tasks
- Increase number of facts in context
- Use longer context passages

### Pack Stacking Issues

If Cocaine doesn't seem to override Morphine:
- Check that both packs are being applied correctly
- Verify pack stacking behavior in `neuromod_tool`
- Consider applying Cocaine with higher intensity

## References

- Original "Stimulant Ceiling" hypothesis from main paper
- Morphine pack: Attention degradation via qk_score_scaling down, temperature reduction, calm bias
- Cocaine pack: QK-score scaling up, temperature reduction, steering vectors (salient)

## Notes

### Morphine Pack Effects

The Morphine pack uses:
- `qk_score_scaling` down (reduces attention sharpness)
- `temperature` down (reduces entropy)
- `style_affect_logit_bias` calm (emotional bias)

This combination should degrade retrieval performance by reducing the model's ability to focus on relevant information in the context. For a stronger "coma" effect, consider:
- Increasing `--intensity-morphine` (e.g., 1.5 or 2.0)
- Creating a custom pack with explicit `kv_decay` or `exponential_decay_kv` effects

