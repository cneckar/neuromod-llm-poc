# Statistics Wiring Fix: Connect Real Mixed-Effects Models

## Problem
The `mixed_effects_model` method in `analysis/statistical_analysis.py` was a stubbed placeholder that just returned an error message, even though a real implementation exists in `neuromod/testing/advanced_statistics.py`.

## Solution

### Changes Made

#### 1. Added Import
```python
from neuromod.testing.advanced_statistics import AdvancedStatisticalAnalyzer
```

#### 2. Initialize in `__init__`
```python
def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
    self.alpha = alpha
    self.n_bootstrap = n_bootstrap
    
    # Initialize advanced statistics analyzer if available
    if ADVANCED_STATS_AVAILABLE:
        self.advanced_stats = AdvancedStatisticalAnalyzer()
    else:
        self.advanced_stats = None
        logger.warning("AdvancedStatisticalAnalyzer not available. Mixed-effects models will not work.")
```

#### 3. Replaced Stubbed Method
The `mixed_effects_model` method now:
- Calls `self.advanced_stats.fit_mixed_effects_model()` with proper parameters
- Auto-detects `group_var` if not provided (tries common names like `prompt_id`, `item_id`, etc.)
- Returns comprehensive results including:
  - `p_values`: Fixed effects p-values
  - `coefficients`: Fixed effects coefficients
  - `coefficients_se`: Standard errors
  - `coefficients_ci`: Confidence intervals
  - `random_effects`: Random effects estimates
  - `aic`, `bic`: Model fit statistics
  - `log_likelihood`: Log-likelihood
  - `n_observations`, `n_groups`: Sample sizes
  - `convergence_warning`: Convergence status
  - `summary`: Model summary string
  - `formula`, `model_name`: Model metadata

### Features

1. **Auto-detection of group_var**: If not specified, tries common column names:
   - `prompt_id`, `item_id`, `seed`, `trial_id`, `replicate_id`, `group`
   - Falls back to categorical columns with reasonable cardinality

2. **Error Handling**: Graceful fallback if:
   - `AdvancedStatisticalAnalyzer` is not available
   - `group_var` cannot be detected
   - Model fitting fails

3. **Comprehensive Results**: Returns all relevant statistics from the mixed-effects model

### Usage

```python
from analysis.statistical_analysis import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()

# Auto-detect group_var
result = analyzer.mixed_effects_model(
    data=df,
    formula="score ~ condition"
)

# Explicit group_var
result = analyzer.mixed_effects_model(
    data=df,
    formula="score ~ condition + pack",
    group_var="prompt_id"
)

if result['success']:
    print(f"AIC: {result['aic']}")
    print(f"P-values: {result['p_values']}")
    print(f"Coefficients: {result['coefficients']}")
else:
    print(f"Error: {result['error']}")
```

### Files Modified

- `analysis/statistical_analysis.py`:
  - Added import for `AdvancedStatisticalAnalyzer`
  - Initialize `advanced_stats` in `__init__`
  - Replaced stubbed `mixed_effects_model` with real implementation

### Impact

- Mixed-effects models now work through the main `StatisticalAnalyzer` interface
- No need to directly import `AdvancedStatisticalAnalyzer` for basic usage
- Consistent API across all statistical methods
- Proper error handling and auto-detection for ease of use

