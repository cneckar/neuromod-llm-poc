# Endpoint Calculation Guide

This guide explains how to use the endpoint calculation system to generate results for Section 4.6 of the experiment execution plan.

## Overview

The endpoint calculation system:
1. Runs relevant psychometric tests with a neuromodulation pack
2. Runs baseline tests (no pack)
3. Runs placebo tests
4. Calculates primary and secondary endpoints by combining subscales
5. Compares treatment vs baseline/placebo
6. Evaluates success criteria
7. Generates JSON reports

## Quick Start

### Basic Usage

Calculate endpoints for a stimulant pack:
```bash
python scripts/calculate_endpoints.py --pack caffeine --model gpt2
```

Calculate endpoints for a psychedelic pack:
```bash
python scripts/calculate_endpoints.py --pack lsd --model gpt2
```

Calculate endpoints for a depressant pack:
```bash
python scripts/calculate_endpoints.py --pack alcohol --model gpt2
```

### Production Models

The script automatically detects production models and uses production mode:
```bash
python scripts/calculate_endpoints.py --pack caffeine --model "meta-llama/Llama-3.1-8B-Instruct"
```

## What Gets Calculated

### Primary Endpoints

**Stimulant Detection** (for: caffeine, cocaine, amphetamine, methylphenidate, modafinil):
- Combines: ADQ-20 (struct + onthread subscales) + PCQ-POP-20 (CLAMP subscale)
- Expected: Increase in focus/structure metrics

**Psychedelic Detection** (for: lsd, psilocybin, dmt, mescaline, 2c_b):
- Combines: PDQ-S (total score) + ADQ-20 (assoc + reroute subscales)
- Expected: Increase in visionary/associative metrics

**Depressant Detection** (for: alcohol, benzodiazepines, heroin, morphine, fentanyl):
- Combines: PCQ-POP-20 (SED subscale) + DDQ (intensity_score)
- Expected: Increase in sedation/calmness metrics

### Secondary Endpoints

- **Cognitive Performance**: CDQ + DDQ + EDQ scores
- **Social Behavior**: SDQ + EDQ scores (for MDMA-like packs)
- **Creativity/Association**: Cognitive tasks creative scores
- **Attention/Focus**: Telemetry attention entropy + cognitive focus
- **Off-target Effects**: Safety monitoring metrics

## Success Criteria

### Primary Endpoints
- ✅ Detection threshold: ≥ 0.5
- ✅ Effect size (Cohen's d): ≥ 0.25
- ✅ P-value: < 0.05
- ✅ Direction: Must match expected (increase/decrease)

### Secondary Endpoints
- ✅ Effect size: ≥ 0.20
- ✅ P-value: < 0.05
- ✅ Direction: Must match expected

## Output

Results are saved to `outputs/endpoints/` with filenames like:
```
endpoints_caffeine_gpt2_20251118_120000.json
```

The JSON file contains:
- Primary endpoint results (treatment, baseline, placebo scores)
- Secondary endpoint results
- Effect sizes and p-values
- Success criteria evaluation
- Overall success status

### Converting to NDJSON for Power Analysis

The power-analysis tooling expects newline-delimited JSON (NDJSON). After
running one or more endpoint calculations, convert all of the summary JSON
files in `outputs/endpoints/` into a single NDJSON file:

```bash
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/pilot_data.jsonl
```

- The script automatically finds every `endpoints_*.json` file in the directory
  and appends its records to the NDJSON file.
- Use `--append` if you want to accumulate into an existing NDJSON file.

You can then pass `outputs/endpoints/pilot_data.jsonl` directly to
`analysis/power_analysis.py --pilot ...`.
## Example Output

```
======================================================================
Calculating Endpoints for Pack: caffeine
Model: gpt2
======================================================================

[*] Running tests for pack: caffeine
[*] Running primary endpoint tests...
  [*] Running ADQ-20...
  [*] Running PCQ-POP-20...
[*] Running secondary endpoint tests...
  [*] Running CDQ, DDQ, EDQ...
  [*] Running cognitive tasks...

[*] Calculating endpoints...

======================================================================
📊 Endpoint Calculation Summary
======================================================================

Pack: caffeine
Model: gpt2
Overall Success: ✅ YES

📈 Primary Endpoints:
  ✅ stimulant_detection:
     Treatment: 0.623
     Baseline: 0.312
     Placebo: 0.298
     Effect Size: 0.856
     P-value: 0.001
     Significant: Yes
     Meets Criteria: Yes

📊 Secondary Endpoints:
  ✅ cognitive_performance:
     Treatment: 0.445
     Baseline: 0.312
     Effect Size: 0.412
  ...
```

## Troubleshooting

**Error: "Test X not found in results"**
- The test may have failed to run
- Check that the pack name is correct
- Verify the model loaded successfully

**Error: "Subscale Y not found"**
- The subscale name may not match
- Check test results structure
- Some tests may not have all subscales

**Low effect sizes**
- This is normal for test models (gpt2)
- Production models (Llama 8B+) should show stronger effects
- Check that the pack is being applied correctly

## Next Steps

After calculating endpoints:
1. Review the JSON output files
2. Compare results across different packs
3. Use results for statistical analysis (Section 4.7)
4. Generate visualizations (Section 5)

