# DialoGPT Scandal Fix: Use Target Model for Optimization

## Problem (FATAL ERROR)

The pack optimizer was hardcoded to use `microsoft/DialoGPT-small` (a 117M parameter, 3-year-old toy model) instead of the actual target model (Llama-3.1-8B-Instruct). This is a **fatal error** because:

1. **Paper Claims vs. Reality**: The paper abstract claims "We benchmark behavioral signatures... on Llama-3.1-8B." But the optimizer code proves packs were tuned on DialoGPT-small and then applied to Llama-3.

2. **Zero Transferability**: Transferability of representation engineering vectors between architectures (GPT-2 style vs. Llama-3 style) is **zero**:
   - Layer dimensions don't match (768 vs 4096)
   - Attention heads are different
   - Architecture differences (GPT-2 vs Llama-3) make vectors meaningless

3. **Garbage Parameters**: If you optimize a pack on DialoGPT, the parameters (weights, layers) will be **meaningless garbage** on Llama-3.

## Solution

### Changes Made

1. **PackOptimizer** (`neuromod/optimization/pack_optimizer.py`):
   - Added `model_name` parameter to `__init__` (default: `"meta-llama/Llama-3.1-8B-Instruct"`)
   - Removed hardcoded `"microsoft/DialoGPT-small"` from `_evaluate_pack()` and `_evaluate_pack_simple()`
   - Now uses `self.model_name` (the target model) for all evaluations
   - Updated convenience functions to accept `model_name` parameter

2. **ProbeEvaluator** (`neuromod/optimization/probe_evaluator.py`):
   - Changed default `model_name` from `"microsoft/DialoGPT-small"` to `"meta-llama/Llama-3.1-8B-Instruct"`
   - Added warnings in docstrings about not using test models for production

3. **DrugDesignLab** (`neuromod/optimization/laboratory.py`):
   - Added `model_name` parameter to `__init__` (default: `"meta-llama/Llama-3.1-8B-Instruct"`)
   - Removed hardcoded `"microsoft/DialoGPT-small"` from `test_pack()` method
   - Now uses `self.model_name` for all model loading

### Key Principle

**CRITICAL**: The model used for optimization **MUST** match the model used for final evaluation. You cannot optimize on one architecture and apply to another.

### Usage

```python
# Correct: Use target model for optimization
optimizer = PackOptimizer(
    model_manager=model_manager,
    evaluation_framework=evaluation_framework,
    config=config,
    model_name="meta-llama/Llama-3.1-8B-Instruct"  # Target model
)

# Wrong: Using test model (will produce garbage on target model)
optimizer = PackOptimizer(
    model_manager=model_manager,
    evaluation_framework=evaluation_framework,
    config=config,
    model_name="microsoft/DialoGPT-small"  # DON'T DO THIS
)
```

### Why This Matters

1. **Architecture Differences**: GPT-2 style (DialoGPT) vs Llama-3 have completely different:
   - Hidden dimensions (768 vs 4096)
   - Layer structures
   - Attention mechanisms
   - Tokenization

2. **Vector Transferability**: Steering vectors optimized on one architecture are meaningless on another:
   - Layer indices don't match
   - Dimensions don't match
   - Attention head structures differ

3. **Scientific Integrity**: The paper claims to benchmark on Llama-3.1-8B, so optimization must also use Llama-3.1-8B.

### Files Modified

- `neuromod/optimization/pack_optimizer.py`:
  - `PackOptimizer.__init__()`: Added `model_name` parameter
  - `PackOptimizer._evaluate_pack()`: Uses `self.model_name` instead of hardcoded DialoGPT
  - `PackOptimizer._evaluate_pack_simple()`: Uses `self.model_name` instead of hardcoded DialoGPT
  - `optimize_pack_for_target()`: Added `model_name` parameter
  - `create_optimized_pack()`: Added `model_name` parameter

- `neuromod/optimization/probe_evaluator.py`:
  - `ProbeEvaluator.evaluate_with_pack()`: Changed default to Llama-3.1-8B-Instruct
  - `evaluate_pack_with_probes()`: Changed default to Llama-3.1-8B-Instruct

- `neuromod/optimization/laboratory.py`:
  - `DrugDesignLab.__init__()`: Added `model_name` parameter
  - `DrugDesignLab.test_pack()`: Uses `self.model_name` instead of hardcoded DialoGPT

### Impact

- **Before**: Packs optimized on DialoGPT-small (117M params, GPT-2 architecture) → garbage on Llama-3.1-8B
- **After**: Packs optimized on Llama-3.1-8B-Instruct (8B params, Llama-3 architecture) → valid for Llama-3.1-8B

This fix ensures that:
1. Optimization and evaluation use the same model
2. Steering vectors are architecture-appropriate
3. Paper claims match actual implementation
4. Scientific integrity is maintained

