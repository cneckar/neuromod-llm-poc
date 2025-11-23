# Statistical Analysis Guide

This guide explains how to perform statistical analysis on endpoint calculation outputs.

## Overview

The statistical analysis system provides:
- **Paired t-tests** and **Wilcoxon signed-rank tests**
- **Benjamini-Hochberg FDR correction** for multiple comparisons
- **Effect size calculations** (Cohen's d, Cliff's delta)
- **Bootstrap confidence intervals**
- **Mixed-effects models** (optional, requires statsmodels)
- **Bayesian hierarchical models** (optional, requires PyMC/ArviZ)

## Data Format

Your endpoint results are stored in JSON files like:
```
outputs/endpoints/endpoints_caffeine_meta-llama_Llama-3.1-8B-Instruct_20251119_163928.json
```

Each file contains:
- `primary_endpoints`: Main detection endpoints (stimulant, psychedelic, depressant)
- `secondary_endpoints`: Additional metrics (cognitive, social, creativity, etc.)
- Each endpoint has: `treatment_score`, `baseline_score`, `placebo_score`, `p_value`, `effect_size`

## Method 1: Using the Statistical Analyzer Directly

### Step 1: Convert Endpoint Files to Analysis Format

First, convert your endpoint JSON files to a format suitable for analysis:

```bash
# Convert all endpoint files in a directory to NDJSON
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/analysis_data.jsonl
```

This creates a flat NDJSON file with one row per endpoint/condition combination.

### Step 2: Load and Analyze Data

Create a Python script to analyze the data:

```python
#!/usr/bin/env python3
"""Run statistical analysis on endpoint data"""

import pandas as pd
import json
from pathlib import Path
from analysis.statistical_analysis import StatisticalAnalyzer

# Load NDJSON data
data = []
with open("outputs/endpoints/analysis_data.jsonl", "r") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Rename columns to match expected format
df = df.rename(columns={
    'pack': 'pack',
    'condition': 'condition',  # 'treatment', 'baseline', 'placebo'
    'endpoint': 'metric',
    'score': 'score',
    'item_id': 'item_id'
})

# Create analyzer
analyzer = StatisticalAnalyzer(alpha=0.05, n_bootstrap=10000)

# Run comprehensive analysis
results = analyzer.analyze_experiment(df)

# Print summary
print(f"Total tests: {results['summary']['total_tests']}")
print(f"Significant tests: {results['summary']['significant_tests']}")
print(f"Significant rate: {results['summary']['significant_rate']:.3f}")

# Print FDR correction info
fdr_info = results['fdr_correction']
print(f"\nFDR Correction:")
print(f"  Tests analyzed: {fdr_info['n_tests']}")
print(f"  Raw significant: {fdr_info['n_significant_raw']}")
print(f"  FDR significant: {fdr_info['n_significant_fdr']}")

# Show significant results
print("\nSignificant Results (FDR corrected):")
for test in results['statistical_tests']:
    if test['significant']:
        print(f"  {test['pack_name']} - {test['metric']}:")
        print(f"    p-value: {test['p_value']:.4f} -> FDR: {test['p_value_fdr']:.4f}")
        print(f"    Effect size ({test['effect_size_type']}): {test['effect_size']:.3f}")
        print(f"    Interpretation: {test['interpretation']}")

# Save results
with open("outputs/analysis/statistical_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
```

## Method 2: Using the Validation Script

The validation script can also analyze real endpoint data:

```bash
# Analyze a specific endpoint file
python scripts/validate_statistics.py \
    --endpoint-file outputs/endpoints/endpoints_caffeine_meta-llama_Llama-3.1-8B-Instruct_20251119_163928.json \
    --output-dir outputs/analysis
```

This will:
1. Load the endpoint file
2. Convert it to analysis format
3. Run all statistical tests
4. Generate a comprehensive report

## Method 3: Analyzing Multiple Packs

To analyze results across multiple packs:

```python
#!/usr/bin/env python3
"""Analyze multiple endpoint files"""

import json
import pandas as pd
from pathlib import Path
from analysis.statistical_analysis import StatisticalAnalyzer

# Collect all endpoint files
endpoint_dir = Path("outputs/endpoints")
endpoint_files = list(endpoint_dir.glob("endpoints_*.json"))

# Load all data
all_data = []
for file in endpoint_files:
    with open(file, "r") as f:
        data = json.load(f)
    
    pack_name = data.get("pack_name")
    model_name = data.get("model_name")
    
    # Extract primary endpoints
    for endpoint_name, endpoint_result in data.get("primary_endpoints", {}).items():
        all_data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name
        })
        all_data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': 'control',
            'condition': 'baseline',
            'metric': endpoint_name,
            'score': endpoint_result.get('baseline_score', 0.0),
            'model': model_name
        })
    
    # Extract secondary endpoints
    for endpoint_name, endpoint_result in data.get("secondary_endpoints", {}).items():
        all_data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': pack_name,
            'condition': 'treatment',
            'metric': endpoint_name,
            'score': endpoint_result.get('treatment_score', 0.0),
            'model': model_name
        })
        all_data.append({
            'item_id': f"{pack_name}_{endpoint_name}",
            'pack': 'control',
            'condition': 'baseline',
            'metric': endpoint_name,
            'score': endpoint_result.get('baseline_score', 0.0),
            'model': model_name
        })

df = pd.DataFrame(all_data)

# Run analysis
analyzer = StatisticalAnalyzer(alpha=0.05)
results = analyzer.analyze_experiment(df)

# Save comprehensive results
output_dir = Path("outputs/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "comprehensive_analysis.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"✅ Analysis complete! Results saved to {output_dir / 'comprehensive_analysis.json'}")
```

## Method 4: Power Analysis

To estimate required sample sizes based on your pilot data:

```bash
# Convert endpoints to NDJSON first
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/pilot_data.jsonl

# Run power analysis
python analysis/power_analysis.py \
    --plan analysis/plan.yaml \
    --pilot outputs/endpoints/pilot_data.jsonl \
    --output outputs/analysis/power_analysis.json
```

## Understanding the Results

### StatisticalResult Fields

Each test result contains:
- `test_name`: "paired_t_test" or "wilcoxon_test"
- `metric`: Endpoint name (e.g., "stimulant_detection")
- `pack_name`: Pack name (e.g., "caffeine")
- `n`: Sample size
- `statistic`: Test statistic value
- `p_value`: Raw p-value (before FDR correction)
- `p_value_fdr`: FDR-corrected p-value
- `effect_size`: Cohen's d or Cliff's delta
- `effect_size_type`: "cohens_d" or "cliffs_delta"
- `ci_lower`, `ci_upper`: Bootstrap confidence interval
- `significant`: Whether result is significant after FDR correction
- `interpretation`: Human-readable interpretation

### Effect Size Interpretation

- **Negligible**: |d| < 0.2
- **Small**: 0.2 ≤ |d| < 0.5
- **Medium**: 0.5 ≤ |d| < 0.8
- **Large**: |d| ≥ 0.8

### FDR Correction

The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR) when testing multiple hypotheses. Results are considered significant if `p_value_fdr < 0.05`.

## Example Workflow

```bash
# 1. Calculate endpoints for multiple packs
python scripts/calculate_endpoints.py --pack caffeine --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack lsd --model "meta-llama/Llama-3.1-8B-Instruct"
python scripts/calculate_endpoints.py --pack alcohol --model "meta-llama/Llama-3.1-8B-Instruct"

# 2. Convert to analysis format
python scripts/export_endpoints_to_ndjson.py \
    --input-dir outputs/endpoints \
    --output outputs/endpoints/analysis_data.jsonl

# 3. Run statistical analysis (using Method 1 script above)
python analyze_endpoints.py

# 4. Review results
cat outputs/analysis/statistical_results.json | python -m json.tool
```

## Advanced: Mixed-Effects Models

For more sophisticated analysis with random effects:

```python
from neuromod.testing.advanced_statistics import run_mixed_effects_analysis

# Load your data as DataFrame
# ...

# Run mixed-effects model
results = run_mixed_effects_analysis(
    data=df,
    formula="score ~ condition + (1|pack) + (1|metric)",
    groups="pack"
)
```

## Troubleshooting

**Error: "No valid p-values found"**
- Check that your endpoint files have non-zero scores
- Verify the data format matches expected structure

**Error: "Insufficient data"**
- Ensure you have at least 2 observations per condition
- Check that both treatment and baseline scores are present

**FDR correction shows 0 significant results**
- This is normal if effects are small or sample size is low
- Check raw p-values to see if any are close to significance
- Consider increasing sample size (more replicates)

## Next Steps

After statistical analysis:
1. Review significant results and effect sizes
2. Generate visualizations (forest plots, effect size plots)
3. Write up results for the paper
4. Consider additional analyses (mixed-effects, Bayesian) if needed

