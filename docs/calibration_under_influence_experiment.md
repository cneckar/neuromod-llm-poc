# Calibration Under the Influence Experiment

## Overview

This experiment tests the hypothesis that **Over-Stimulation (Focus) leads to Overfitting Errors (Brittleness/Overconfidence)** by measuring the model's calibration and out-of-distribution (OOD) performance under varying "dosages" of stimulant packs.

## Hypothesis

Increasing "Stimulation" (via QK-Scaling and Temperature reduction) will:
- ✅ Decrease entropy (as shown in the paper)
- ❌ Increase Expected Calibration Error (ECE) 
- ❌ Degrade OOD Generalization

The "Over-Stimulated" model will be hyper-confident (high probability) but equally wrong as the baseline, leading to a massive spike in ECE. It "hallucinates confidence."

## Experimental Design

### Step 1: Graded Stimulant Packs

We create a graded series of Stimulant Packs that progressively increase QK-Score Scaling (sharpening attention) and lower Temperature:

- **`cocaine_10`**: Low-dose stimulant (10% intensity)
- **`cocaine_50`**: Medium-dose stimulant (50% intensity)  
- **`cocaine_100`**: High-dose stimulant (100% intensity - maximum over-stimulation)

**Control**: Base Model with `none` pack (no neuromodulation)

**Variable**: Stimulation Intensity ($I \in [0.0, 1.0]$)

### Step 2: Measure Overconfidence (Calibration Error)

Run the MMLU (Massive Multitask Language Understanding) benchmark, capturing the logits of the answers (A, B, C, D) before sampling.

**Metric**: Expected Calibration Error (ECE)

**Question**: When the model assigns 99% probability to token "A", is it correct 99% of the time?

**Prediction**: The "Over-Stimulated" model will be hyper-confident (high prob) but equally wrong as the baseline, leading to a massive spike in ECE.

### Step 3: Measure Brittleness (Adversarial Overfitting)

Test the model on Out-of-Distribution (OOD) tasks that require "soft" logic:
- Creative writing prompts
- Riddles
- Ambiguous ethical scenarios

**Metrics**:
- **Perplexity**: Higher perplexity indicates less confident, more exploratory responses
- **Diversity (Unique n-grams)**: Measures mode collapse vs. creative diversity
- **Repetition Rate**: Higher repetition indicates brittleness and overfitting

**Prediction**: The "Over-Stimulated" model will suffer from Mode Collapse, repeating generic phrases ("It is important to note...") or refusing to engage with ambiguity, whereas the "Sober" or "Psychedelic" models will handle the distribution shift more gracefully.

### Step 4: Compare Base vs. RLHF

Run the same battery on:
- **Llama-3-Base (Un-tuned)** + Stimulant Pack
- **Llama-3-Instruct (RLHF)**

**Hypothesis**: If the "Stimulant Ceiling" theory is correct, the Base Model + Stimulant should exhibit the same high ECE/Brittleness profile as the raw Instruct Model. This would prove that RLHF is functionally identical to "over-fitting" the attention mechanism.

## Usage

### Basic Usage

```bash
python scripts/calibration_under_influence_experiment.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --packs none cocaine_10 cocaine_50 cocaine_100 \
  --max-questions 100
```

### Compare Base vs. RLHF

```bash
# Test Base Model
python scripts/calibration_under_influence_experiment.py \
  --model meta-llama/Llama-3.1-8B \
  --packs none cocaine_100 \
  --max-questions 100

# Test Instruct Model
python scripts/calibration_under_influence_experiment.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --packs none cocaine_100 \
  --max-questions 100
```

### Custom Packs and Intensities

```bash
python scripts/calibration_under_influence_experiment.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --packs none caffeine cocaine amphetamine \
  --intensities 1.0 0.5 0.75 1.0 \
  --max-questions 50
```

## Output

The experiment generates two output files:

1. **`calibration_experiment_<timestamp>.json`**: Complete results in JSON format
2. **`calibration_summary_<timestamp>.txt`**: Human-readable summary

### Metrics Explained

#### Calibration Metrics

- **ECE (Expected Calibration Error)**: Average absolute difference between confidence and accuracy across bins. Lower is better. High ECE indicates overconfidence.
- **MCE (Maximum Calibration Error)**: Maximum difference in any single bin. Measures worst-case calibration.
- **Brier Score**: Mean squared error between confidence and accuracy. Lower is better.
- **Accuracy**: Overall correctness on MMLU questions.
- **Confidence**: Average predicted probability of the chosen answer.

#### OOD Metrics

- **Perplexity**: Average perplexity on OOD tasks. Higher indicates less confident, more exploratory responses.
- **Diversity Ratio**: Unique n-grams / Total n-grams. Higher indicates more creative, less repetitive responses.
- **Repetition Rate**: Proportion of repeated bigrams. Lower is better (indicates less mode collapse).

## Expected Results

Based on the hypothesis:

1. **ECE should increase** with stimulation intensity:
   - `none`: Low ECE (well-calibrated)
   - `cocaine_10`: Slightly higher ECE
   - `cocaine_50`: Moderate ECE increase
   - `cocaine_100`: High ECE (overconfident)

2. **OOD Diversity should decrease** with stimulation:
   - `none`: High diversity, creative responses
   - `cocaine_100`: Low diversity, repetitive/generic responses

3. **Base + Stimulant ≈ RLHF**:
   - Base model with `cocaine_100` should show similar ECE/brittleness as Instruct model with `none`

## Interpretation

If the hypothesis is confirmed:

- **Over-stimulation = Overfitting**: High QK-scaling and low temperature create a "focus trap" that reduces entropy but increases brittleness.
- **RLHF as Stimulation**: RLHF training may be functionally equivalent to applying a stimulant pack, explaining why RLHF models are often overconfident and brittle.
- **The Stimulant Ceiling**: There's a limit to how much "focus" improves performance before it degrades generalization.

## Dependencies

- `datasets` library (for MMLU): `pip install datasets`
- If `datasets` is not available, the script will use a small synthetic dataset for testing

## References

- MMLU Dataset: [Hendrycks et al., 2021](https://arxiv.org/abs/2009.03300)
- Expected Calibration Error: [Guo et al., 2017](https://arxiv.org/abs/1706.04599)

