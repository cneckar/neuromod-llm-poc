# Model Validation Priority Guide

## Primary Models for Paper (Section 4.1)

Based on the paper outline and execution plan, you should validate these three primary models:

### ✅ 1. Qwen/Qwen2.5-Omni-7B
**Status**: ✅ **VALIDATED**  
**Size**: 7B parameters  
**Quantization**: 8-bit  
**Authentication**: May be required  
**VRAM Required**: ~8-12GB  
**Loading Time**: ~50-60 seconds (cached)

---

### 2. meta-llama/Llama-3.1-70B-Instruct
**Status**: ⏳ **PENDING**  
**Size**: 70B parameters  
**Quantization**: 4-bit (required)  
**Authentication**: ✅ **REQUIRED** (HuggingFace token)  
**VRAM Required**: ~40GB  
**Expected Loading Time**: 5-15 minutes (cached)

**To Validate**:
```bash
# Set HuggingFace token first
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Run validation
python scripts/validate_models.py --model "meta-llama/Llama-3.1-70B-Instruct"
```

**Prerequisites**:
1. Accept license at: https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct
2. Get token from: https://huggingface.co/settings/tokens
3. Set environment variable with token

---

### 3. mistralai/Mixtral-8x22B-Instruct-v0.1
**Status**: ⏳ **PENDING**  
**Size**: 8×22B parameters (Mixture of Experts)  
**Quantization**: 4-bit (required)  
**Authentication**: ✅ **REQUIRED** (HuggingFace token)  
**VRAM Required**: ~45-50GB  
**Expected Loading Time**: 10-20 minutes (cached)

**To Validate**:
```bash
# With token already set
python scripts/validate_models.py --model "mistralai/Mixtral-8x22B-Instruct-v0.1"
```

**Prerequisites**:
1. Accept license at: https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1
2. Get token from: https://huggingface.co/settings/tokens
3. Set environment variable with token

---

## Recommended Validation Order

### Option A: Validate All Three (Complete Section 4.1)
```bash
# Set token once
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Validate all three
python scripts/validate_models.py
```

This will test:
1. ✅ Qwen/Qwen2.5-Omni-7B (already done)
2. ⏳ meta-llama/Llama-3.1-70B-Instruct
3. ⏳ mistralai/Mixtral-8x22B-Instruct-v0.1

**Time Estimate**: 15-35 minutes total (if models are cached)

---

### Option B: Validate Individual Models (If You Have Limited Resources)

If you don't have enough GPU memory or authentication for all models, prioritize:

1. **Qwen/Qwen2.5-Omni-7B** ✅ (Already done - smallest, fastest)
2. **meta-llama/Llama-3.1-70B-Instruct** (Most important for paper, requires auth)
3. **mistralai/Mixtral-8x22B-Instruct-v0.1** (Largest, requires most resources)

---

## Alternative Models (If Primary Models Fail)

If the primary models are unavailable or require too many resources, these alternatives are configured:

### Smaller Alternatives:
- `meta-llama/Llama-3.1-8B-Instruct` (8B, requires auth, ~8-12GB VRAM)
- `Qwen/Qwen-2.5-7B` (7B, may require auth, ~8-12GB VRAM)
- `Qwen/Qwen2.5-32B-Instruct` (32B, requires auth, ~20-24GB VRAM with 4-bit)

### Test Models (For Framework Validation):
- `gpt2` (124M, no auth, ~1GB RAM, test mode only)
- `distilgpt2` (82M, no auth, ~500MB RAM, test mode only)

---

## Validation Checklist

- [x] Qwen/Qwen2.5-Omni-7B - ✅ Validated
- [ ] meta-llama/Llama-3.1-70B-Instruct - Requires auth + ~40GB VRAM
- [ ] mistralai/Mixtral-8x22B-Instruct-v0.1 - Requires auth + ~45-50GB VRAM

---

## Next Steps After Validation

Once all three primary models are validated:

1. ✅ Section 4.1 action items will be complete
2. Review generated documentation in `outputs/validation/models/`
3. Proceed to Section 4.2 (Neuromodulation Packs Implementation)

---

## Quick Commands

```bash
# Validate Llama-3.1-70B (requires auth)
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
python scripts/validate_models.py --model "meta-llama/Llama-3.1-70B-Instruct"

# Validate Mixtral-8x22B (requires auth)
python scripts/validate_models.py --model "mistralai/Mixtral-8x22B-Instruct-v0.1"

# Validate all three at once
python scripts/validate_models.py
```

