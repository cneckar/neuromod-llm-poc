# Reproduction Guide: The Golden Path

This guide provides step-by-step instructions for reproducing the experiments described in the paper. This is the "Golden Path" for reviewers and researchers who want to replicate our results.

## Overview

This guide merges the pilot study plan and experiment execution plan into a single, streamlined workflow. Follow these steps in order to reproduce the complete experimental pipeline.

## Prerequisites

### System Requirements
- **Python 3.8+**
- **GPU Recommended**: For models 7B+ parameters
- **Memory**: 16GB+ RAM, 24GB+ VRAM for 70B models
- **Storage**: 50GB+ free space for model downloads

### Software Setup

```bash
# Clone repository
git clone https://github.com/cneckar/neuromod-llm-poc.git
cd neuromod-llm-poc

# Create virtual environment
python -m venv venv --system-site-packages
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt --no-deps
pip install -e .

# Set up Hugging Face credentials (required for Llama models)
hf auth login
```

See [Troubleshooting](troubleshooting.md) for detailed setup help.

## Phase 1: Validation (Day 1)

Before running experiments, validate that all components work correctly.

### 1.1 Model Validation

```bash
# Validate model loading (test model)
python scripts/validate_models.py --model gpt2 --test-mode

# Validate production models (requires HF auth)
python scripts/validate_models.py --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/validate_models.py --model "meta-llama/Llama-3.1-8B"

python scripts/validate_models.py --model "meta-llama/Meta-Llama-3.1-70B-Instruct"
python scripts/validate_models.py --model "meta-llama/Meta-Llama-3.1-70B"
```

**Expected Output**: Model loads successfully, generation test passes.

**Validation Results**:
- ✅ Llama-3.1-70B-Instruct: Loads in ~40 minutes
- ✅ Qwen-2.5-Omni-7B: Loads in ~32 seconds
- ⚠️ Mixtral-8×22B: Requires more GPU memory than available on test system
- 🧪 openai/gpt-oss-20b: Supported via 4-bit loading (expect >48GB VRAM or multi-GPU)
- 🧪 openai/gpt-oss-120b: Research-only tier, assumes >80GB VRAM or distributed setup

### 1.2 Pack Validation

```bash
# Validate all packs
python scripts/validate_packs.py
```

**Expected Output**: All 31 packs load successfully, all effects validated.

### 1.3 Blinding Verification

```bash
# Audit all test prompts for blinding issues
python scripts/audit_blinding.py
```

**Expected Output**: 0 leakage issues found, all prompts are generic.

### 1.4 Benchmark Validation

```bash
# Validate all benchmarks on test model
python scripts/validate_benchmarks.py --model gpt2 --test-mode

# Validate on production model
python scripts/validate_benchmarks.py --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/validate_benchmarks.py --model "meta-llama/Llama-3.1-8B"

python scripts/validate_benchmarks.py --model "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
python scripts/validate_benchmarks.py --model "Llama-4-Maverick-17B-128E"
```

**Expected Output**: All psychometric tests, cognitive tasks, telemetry, and safety monitoring pass.

## Phase 2: Data Collection

### 2.1 The single reproduction path — `scripts/reproduce.py`

Reproduction is consolidated into **one tiered playbook**. It regenerates the paper's text
**and** visual collateral and writes `outputs/reproduction/REPRODUCTION_REPORT.md` mapping each
artifact to the claim it supports. Full tier table + artifact→claim map: [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

```bash
# Tier 0 — CPU only: validators + committed-data figures + dry-run visual pipeline
python scripts/reproduce.py --tier 0

# Tier 1 — one GPU: real SDXL dose-response + the text battery on ungated gpt2
python scripts/reproduce.py --tier 1 --seeds 16

# Tier 2 — gated Llama-3.1-8B at paper scale (the only tier with the paper's exact numbers)
HUGGINGFACE_TOKEN=hf_... python scripts/reproduce.py --tier 2

# Inspect the plan without running anything
python scripts/reproduce.py --tier 2 --list
```

Three ways to run, one path:
1. **Locally via the script** — the commands above.
2. **Locally via the notebook** — open `notebooks/reproduce_paper_colab.ipynb` in Jupyter (it
   auto-detects that you're already in the repo and skips the clone/install cells).
3. **In Colab** — open the same notebook, pick a tier, *Run all*.

> The legacy `python reproduce_results.py [--test-mode]` command still works but now simply
> **forwards** to `scripts/reproduce.py` (`--test-mode` → `--tier 1`, default → `--tier 2`).

The power/NDJSON steps below run automatically as the `export_ndjson` / `power_analysis` stages;
you can also run them standalone:

```bash
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/reproduction/endpoints \
    --output outputs/reproduction/endpoints/pilot_data.jsonl
python analysis/power_analysis.py \
    --plan analysis/plan.yaml \
    --pilot outputs/reproduction/endpoints/pilot_data.jsonl
```

**Expected Output**: Pilot data confirms n≥126 per condition is sufficient for power.

### 2.2 Primary Endpoint Collection

Run endpoint calculations for all primary packs on the target model:

#### Stimulant Detection Packs
```bash
python scripts/calculate_endpoints.py --pack caffeine --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack cocaine --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack amphetamine --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack methylphenidate --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack modafinil --model "meta-llama/Llama-3.1-8B-Instruct"
```

#### Psychedelic Detection Packs
```bash
python scripts/calculate_endpoints.py --pack lsd --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack psilocybin --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack dmt --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack mescaline --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack 2c_b --model "meta-llama/Llama-3.1-8B-Instruct"
```

#### Depressant Detection Packs
```bash
python scripts/calculate_endpoints.py --pack alcohol --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack benzodiazepines --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack heroin --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack morphine --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack fentanyl --model "meta-llama/Llama-3.1-8B-Instruct"
```

**Note**: Each pack run requires n≥126 items per condition. Use `--skip-completed` to resume interrupted runs.

**Output Location**: `outputs/endpoints/endpoints_<pack>_<model>_<timestamp>.json`

#### Understanding Endpoint Calculation

The endpoint calculation system:
1. Runs relevant psychometric tests with a neuromodulation pack
2. Runs baseline tests (no pack)
3. Runs placebo tests
4. Calculates primary and secondary endpoints by combining subscales
5. Compares treatment vs baseline/placebo
6. Evaluates success criteria
7. Generates JSON reports

**Primary Endpoints**:
- **Stimulant Detection** (caffeine, cocaine, amphetamine, methylphenidate, modafinil): Combines ADQ-20 (struct + onthread subscales) + PCQ-POP-20 (CLAMP subscale). Expected: Increase in focus/structure metrics.
- **Psychedelic Detection** (lsd, psilocybin, dmt, mescaline, 2c_b): Combines PDQ-S (total score) + ADQ-20 (assoc + reroute subscales). Expected: Increase in visionary/associative metrics.
- **Depressant Detection** (alcohol, benzodiazepines, heroin, morphine, fentanyl): Combines PCQ-POP-20 (SED subscale) + DDQ (intensity_score). Expected: Increase in sedation/calmness metrics.

**Secondary Endpoints**:
- Cognitive Performance: CDQ + DDQ + EDQ scores
- Social Behavior: SDQ + EDQ scores (for MDMA-like packs)
- Creativity/Association: Cognitive tasks creative scores
- Attention/Focus: Telemetry attention entropy + cognitive focus
- Off-target Effects: Safety monitoring metrics

**Success Criteria**:
- Primary endpoints: Detection threshold ≥ 0.5, Effect size (Cohen's d) ≥ 0.25, P-value < 0.05, Direction matches expected
- Secondary endpoints: Effect size ≥ 0.20, P-value < 0.05, Direction matches expected

**Converting to NDJSON for Power Analysis**:
```bash
# Convert all endpoint files to NDJSON format
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/pilot_data.jsonl
```

The script automatically finds every `endpoints_*.json` file in the directory and appends its records to the NDJSON file. Use `--append` if you want to accumulate into an existing NDJSON file.

### 2.3 Emotion Signature Validation (Optional but Recommended)

The story emotion test provides qualitative validation of pack effects by tracking emotional shifts in generated narratives. This complements the quantitative psychometric tests and helps verify that packs produce expected emotional signatures.

#### Representative Pack Tests

**1. The Psychedelic Representative (LSD)**
```bash
# Expectation: High Surprise, Fear (Volatility), Joy (Euphoria)
python -m neuromod.testing.story_emotion_test --model "meta-llama/Llama-3.1-8B-Instruct" --pack lsd
```

**2. The Depressant Representative (Morphine)**
```bash
# Expectation: Low Arousal, High Trust (Passive), Low Anger
python -m neuromod.testing.story_emotion_test --model "meta-llama/Llama-3.1-8B-Instruct" --pack morphine
```

**3. The Stimulant Representative (Caffeine)**
```bash
# Expectation: High Anticipation (Drive), High Joy, potentially Anger (Agitation)
# Even if detection failed, we want to see if the *tone* shifted.
python -m neuromod.testing.story_emotion_test --model "meta-llama/Llama-3.1-8B-Instruct" --pack caffeine
```

**4. The Baseline (Placebo)**
```bash
# Expectation: Balanced, neutral profile (The control)
python -m neuromod.testing.story_emotion_test --model "meta-llama/Llama-3.1-8B-Instruct" --pack placebo
```

**Expected Output**: Emotion tracking results showing shifts in joy, sadness, anger, fear, surprise, disgust, trust, and anticipation across the narrative. The LSD pack should show high volatility (fear/surprise) and euphoria (joy), while morphine should show low arousal and high trust. Caffeine may show increased anticipation and drive, even if quantitative detection failed.

**Output Location**: `outputs/reports/emotion/emotion_results_story_emotion_test_*.json`

### 2.4 Convert to Analysis Format

```bash
# Convert all endpoint files to NDJSON for statistical analysis
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/experiment_data.jsonl
```

## Phase 3: Statistical Analysis

### 3.1 Run Statistical Analysis

```bash
# Analyze all endpoint results
python scripts/analyze_endpoints.py \
    --input-dir outputs/endpoints \
    --output outputs/analysis/statistical_results.json
```

**Output**: Statistical results with p-values, effect sizes, FDR correction, and significance flags.

### 3.2 Power Analysis

```bash
# Verify sample sizes are sufficient
python analysis/power_analysis.py \
    --plan analysis/plan.yaml \
    --pilot outputs/endpoints/experiment_data.jsonl
```

**Expected Output**: Confirms n≥126 per condition achieves 80% power for d≥0.25.

## Phase 4: Validation Checklist

Before considering reproduction complete, verify:

- [ ] All models load successfully
- [ ] All packs validate and apply correctly
- [ ] All psychometric tests complete without errors
- [ ] All cognitive tasks complete without errors
- [ ] All telemetry metrics collected successfully
- [ ] Endpoint calculations complete for all primary packs
- [ ] Statistical analysis completes successfully
- [ ] Effect sizes calculated correctly
- [ ] FDR correction applied
- [ ] Power analysis confirms sufficient sample size

## Expected Outputs

### Data Files
```
outputs/endpoints/
├── endpoints_<pack>_<model>_<timestamp>.json  # Endpoint results per pack
└── experiment_data.jsonl                       # Combined NDJSON for analysis
```

### Analysis Files
```
outputs/analysis/
├── statistical_results.json                    # Statistical analysis results
└── power_analysis_report.json                  # Power analysis results
```

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

## Time Estimates

- **Phase 1 (Validation)**: 2-4 hours
- **Phase 2 (Data Collection)**: 2-3 days per pack (15 packs = 30-45 days)
- **Phase 3 (Analysis)**: 1-2 hours
- **Total**: ~30-50 days for complete reproduction (can be parallelized)

## Next Steps

After reproduction:
1. Compare your results to published results
2. Verify statistical significance matches
3. Check effect sizes are in expected ranges
4. Review any discrepancies with the methodology

For detailed methodology explanations, see [Methodology Guide](methodology.md).

