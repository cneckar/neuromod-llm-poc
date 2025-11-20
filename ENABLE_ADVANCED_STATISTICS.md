# Enabling Mixed-Effects and Bayesian Models

This guide walks through enabling the two optional advanced statistical models in Section 4.7.

## Current Status

- ✅ **PyMC/ArviZ installed**: Confirmed (PyMC 5.26.1, ArviZ 0.22.0)
- ⚠️ **Mixed-effects models**: Data format issue with categorical variables
- ⚠️ **Bayesian models**: Detection/import issue

## Issue 1: Mixed-Effects Models

### Problem
The `statsmodels.mixedlm` function has trouble parsing formulas with categorical string variables, especially in the random effects part `(1|prompt_set)`.

### Solution
We need to ensure:
1. **Group variable** (`prompt_set`) is properly formatted as string
2. **Predictor variables** in the formula are numeric (not categorical strings)

### Fix Applied
- Modified `fit_mixed_effects_model()` to convert group variables to strings
- Modified validation script to convert categorical predictors to numeric before fitting

### Testing
Run the validation script:
```bash
python scripts/validate_statistics.py --mock-only
```

## Issue 2: Bayesian Models

### Problem
The function returns `None` even though PyMC/ArviZ are installed. The `BAYESIAN_AVAILABLE` flag is `True`, but the function still returns `None`.

### Solution
The issue was in the function logic - it was checking `BAYESIAN_AVAILABLE` correctly, but there may be an import issue at runtime.

### Fix Applied
- Simplified the Bayesian availability check
- Added explicit imports inside the function to ensure they're available at runtime
- Improved error handling to show actual import errors

### Testing
Run the validation script:
```bash
python scripts/validate_statistics.py --mock-only
```

## Step-by-Step Fix

### 1. Fix Mixed-Effects Model Data Format

The issue is that `statsmodels.mixedlm` needs:
- Group variables as strings (already handled)
- Predictor variables as numeric (needs conversion)

**File**: `neuromod/testing/advanced_statistics.py`

Already fixed in `fit_mixed_effects_model()`:
- Converts group variable to string
- Formula should use numeric predictors

**File**: `scripts/validate_statistics.py`

Already fixed:
- Converts `condition` to `condition_numeric` before fitting
- Uses numeric formula: `score ~ condition_numeric + (1|prompt_set)`

### 2. Fix Bayesian Model Detection

**File**: `neuromod/testing/advanced_statistics.py`

Already fixed:
- Simplified availability check
- Added explicit imports inside function
- Better error messages

### 3. Verify Installation

Check that PyMC and ArviZ are properly installed:
```bash
python -c "import pymc as pm; import arviz as az; print('PyMC:', pm.__version__); print('ArviZ:', az.__version__)"
```

### 4. Run Validation

```bash
python scripts/validate_statistics.py --mock-only
```

Expected output:
- Mixed-effects model: Should fit successfully
- Bayesian model: Should fit successfully (may take a minute for sampling)

## Troubleshooting

### Mixed-Effects Model Still Fails

If you see: `Error evaluating factor: TypeError: Cannot perform 'ror_' with a dtyped [object] array`

**Solution**: Ensure all variables in the formula are numeric:
```python
# Convert categorical to numeric
data['condition_numeric'] = pd.Categorical(data['condition']).codes
formula = "score ~ condition_numeric + (1|prompt_set)"
```

### Bayesian Model Returns None

If you see: `Bayesian analysis returned None`

**Check**:
1. PyMC/ArviZ are installed: `pip list | grep -E "pymc|arviz"`
2. Import works: `python -c "import pymc as pm; import arviz as az"`
3. Check the actual error in the traceback

**Solution**: The function should now show the actual error if there is one.

### Performance Warning

You may see: `g++ not detected! PyTensor will be unable to compile C-implementations`

This is a **warning, not an error**. The Bayesian model will still work, just slower. To fix:
- Install a C++ compiler (e.g., `conda install gxx` or Visual Studio Build Tools on Windows)
- Or set: `export PYTENSOR_FLAGS=cxx=""` to suppress the warning

## Expected Results

After fixes, you should see:

```
[*] Testing mixed-effects model...
[OK] Mixed-effects model fitted
     AIC: XXX.XXX, BIC: XXX.XXX
     Fixed effects: {...}

[*] Testing Bayesian hierarchical model...
[OK] Bayesian model fitted
     WAIC: XXX.XXX, LOO: XXX.XXX
```

## Next Steps

Once both models are working:

1. Update `EXPERIMENT_EXECUTION_PLAN.md` Section 4.7:
   - Mark mixed-effects models as ✅ Available
   - Mark Bayesian models as ✅ Available

2. Test with real endpoint data:
   ```bash
   python scripts/validate_statistics.py --endpoint-file outputs/endpoints/endpoints_caffeine_gpt2_*.json
   ```

3. Use in actual analysis:
   - Mixed-effects models for accounting for prompt-level variation
   - Bayesian models for hierarchical inference with credible intervals

