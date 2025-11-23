# PackOptimizer Refactor: Remove Hardcoded DialoGPT

## Problem

PackOptimizer had `model_name` as a separate parameter, which could lead to inconsistencies. The user requested to refactor it to use `OptimizationConfig.model_name` instead.

## Solution

### Changes Made

1. **OptimizationConfig** (`neuromod/optimization/pack_optimizer.py`):
   - Added `model_name: str = "meta-llama/Llama-3.1-8B-Instruct"` field to `OptimizationConfig` dataclass
   - This centralizes model selection in the configuration

2. **PackOptimizer.__init__**:
   - Removed `model_name` parameter from `__init__`
   - Now gets `model_name` from `self.config.model_name`
   - Updated docstring to reflect that model_name should be set in config

3. **Convenience Functions**:
   - `optimize_pack_for_target()`: Still accepts `model_name` parameter, but now passes it to `OptimizationConfig`
   - `create_optimized_pack()`: Unchanged (still accepts `model_name` and passes to `optimize_pack_for_target`)

4. **DrugDesignLab** (`neuromod/optimization/laboratory.py`):
   - Updated to create `OptimizationConfig` with `model_name` instead of passing it separately

### Benefits

1. **Centralized Configuration**: Model name is now part of the optimization configuration
2. **Consistency**: All optimization settings (including model) are in one place
3. **No Hardcoded Models**: Removed any remaining hardcoded DialoGPT references
4. **Better API**: Configuration is explicit and clear

### Usage

**Before**:
```python
optimizer = PackOptimizer(
    model_manager=model_manager,
    evaluation_framework=evaluation_framework,
    config=config,
    model_name="meta-llama/Llama-3.1-8B-Instruct"  # Separate parameter
)
```

**After**:
```python
config = OptimizationConfig(model_name="meta-llama/Llama-3.1-8B-Instruct")
optimizer = PackOptimizer(
    model_manager=model_manager,
    evaluation_framework=evaluation_framework,
    config=config  # model_name is in config
)
```

### Files Modified

- `neuromod/optimization/pack_optimizer.py`:
  - `OptimizationConfig`: Added `model_name` field
  - `PackOptimizer.__init__()`: Removed `model_name` parameter, uses `config.model_name`
  - `optimize_pack_for_target()`: Passes `model_name` to `OptimizationConfig`

- `neuromod/optimization/laboratory.py`:
  - `DrugDesignLab.__init__()`: Creates `OptimizationConfig` with `model_name`

### Verification

- ✅ No hardcoded DialoGPT references remain
- ✅ Model name is configurable via `OptimizationConfig`
- ✅ Backward compatibility maintained (convenience functions still accept `model_name`)
- ✅ All code paths use `config.model_name`

