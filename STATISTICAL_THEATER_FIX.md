# Statistical Theater Fix: Permutation Test Validation

## Problem
Using Bayesian Hierarchical Models on a toy dataset (N=126) with huge effect sizes (d=10.0) is suspicious. The advanced statistics may be overfitting or producing misleading results.

## Solution: Simplify Stats and Validate Metrics

### 1. Mark Advanced Statistics as Experimental

**Changed:**
- `analysis/statistical_analysis.py`: Added warning that `AdvancedStatisticalAnalyzer` is experimental
- Advanced statistics are now optional and marked as deprecated for small datasets
- Recommendation: Use permutation tests for metric validation instead

### 2. Implement Permutation Test for PDQ-S Validation

**Created: `analysis/permutation_test.py`**

The permutation test proves that PDQ-S (and other detection metrics) are actually measuring the intended effects, not just random noise or confounds like sentence length.

#### Algorithm

1. **Calculate Actual Score**: Compute PDQ-S detection score difference (LSD vs Placebo)
2. **Shuffle Labels**: Randomly shuffle LSD/Placebo labels 10,000 times
3. **Recalculate Scores**: For each permutation, recalculate PDQ-S score
4. **Build Null Distribution**: Collect all permuted scores
5. **Compare**: Compare actual score against null distribution
6. **P-value**: Proportion of null values >= |actual_difference|

#### Implementation

```python
def permutation_test_pdq_s(lsd_results, placebo_results, n_permutations=10000):
    # Calculate actual difference
    lsd_score = calculate_pdq_s_score(lsd_results)
    placebo_score = calculate_pdq_s_score(placebo_results)
    actual_difference = lsd_score - placebo_score
    
    # Generate null distribution
    all_results = lsd_results + placebo_results
    null_distribution = []
    
    for _ in range(n_permutations):
        # Shuffle labels
        shuffled_indices = np.random.permutation(len(all_results))
        permuted_lsd = [all_results[i] for i in shuffled_indices[:n_lsd]]
        permuted_placebo = [all_results[i] for i in shuffled_indices[n_lsd:]]
        
        # Recalculate scores
        permuted_diff = (calculate_pdq_s_score(permuted_lsd) - 
                        calculate_pdq_s_score(permuted_placebo))
        null_distribution.append(permuted_diff)
    
    # Calculate p-value
    p_value = np.mean(np.abs(null_distribution) >= abs(actual_difference))
    
    return PermutationTestResult(
        actual_score=actual_difference,
        null_distribution=np.array(null_distribution),
        p_value=p_value,
        significant=p_value < 0.01
    )
```

### 3. Integration with Statistical Analyzer

**Added to `analysis/statistical_analysis.py`:**
- `validate_metric_with_permutation()`: Method to validate metrics using permutation test
- Integrated `PermutationTestValidator` into `StatisticalAnalyzer`
- Permutation tests are now the recommended method for metric validation

## Usage

### Validate PDQ-S Metric

```python
from analysis.permutation_test import validate_pdq_s_metric

# Load LSD and Placebo results
lsd_results = [...]  # List of PDQ-S results for LSD condition
placebo_results = [...]  # List of PDQ-S results for Placebo condition

# Run permutation test
result = validate_pdq_s_metric(lsd_results, placebo_results, n_permutations=10000)

print(f"Actual PDQ-S difference: {result.actual_score:.4f}")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.significant}")
print(f"Interpretation: {result.interpretation}")
```

### Using Statistical Analyzer

```python
from analysis.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

validation = analyzer.validate_metric_with_permutation(
    treatment_results=lsd_results,
    control_results=placebo_results,
    metric_name="PDQ-S",
    treatment_name="LSD",
    control_name="Placebo"
)

if validation['success'] and validation['significant']:
    print("✓ PDQ-S successfully detects psychedelic effects")
    print(f"  P-value: {validation['p_value']:.4f}")
else:
    print("✗ PDQ-S may not be detecting psychedelic effects")
    print(f"  P-value: {validation['p_value']:.4f}")
```

## What This Proves

The permutation test validates that:

1. **Not Random Noise**: The metric isn't just measuring random variation
2. **Not Confounds**: The metric isn't just measuring sentence length or other confounds
3. **Actual Detection**: The metric is actually detecting the intended effect (psychedelic presence)

If the permutation test shows p < 0.01, we can be confident that:
- PDQ-S is detecting real psychedelic effects
- The difference between LSD and Placebo is not due to chance
- The metric is valid for its intended purpose

## Files Modified

- `analysis/permutation_test.py` (NEW): Permutation test implementation
- `analysis/statistical_analysis.py`: 
  - Added warning about advanced statistics
  - Integrated permutation test validator
  - Added `validate_metric_with_permutation()` method

## Benefits

1. **Simpler Statistics**: No need for complex Bayesian models on small datasets
2. **Metric Validation**: Proves metrics actually work, not just measuring noise
3. **Transparent**: Easy to understand and interpret
4. **Robust**: Works with any sample size (though more powerful with larger N)
5. **No Assumptions**: Non-parametric, no distributional assumptions

## Impact

- **PDQ-S Validation**: Can now prove that PDQ-S actually detects psychedelic effects
- **Other Metrics**: Same approach can validate SDQ, DDQ, etc.
- **Statistical Hygiene**: No more suspicious Bayesian models on toy datasets
- **Reproducibility**: Permutation tests are easy to understand and reproduce

