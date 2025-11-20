# Section 4.1 Completion Commands

## Overview

To complete all action items in Section 4.1 (Models and Serving Stacks), you need to:

1. ✅ Model support system implemented (already done)
2. ✅ Quantization support (already done)
3. ⏳ Test all three primary models load successfully
4. ⏳ Document model loading times and memory usage (auto-generated when tests run)

---

## Commands to Run

### Option 1: Test All Three Primary Models (Recommended)

This will test all three models and generate complete documentation:

```bash
python scripts/validate_models.py
```

**Expected Output:**
- Tests `meta-llama/Llama-3.1-70B-Instruct` (requires auth)
- Tests `Qwen/Qwen2.5-Omni-7B` (may require auth)
- Tests `mistralai/Mixtral-8x22B-Instruct-v0.1` (requires auth)
- Generates JSON results file
- Generates Markdown summary with loading times and memory usage

**Note**: This will take a long time (potentially hours) if models need to be downloaded. Models that require authentication will be skipped with clear error messages.

---

### Option 2: Test Individual Models

Test models one at a time:

#### Test Qwen2.5-Omni-7B (if available without auth):
```bash
python scripts/validate_models.py --model "Qwen/Qwen2.5-Omni-7B"
```

#### Test Llama-3.1-70B-Instruct (requires HuggingFace token):
```bash
# First set your token (Windows PowerShell):
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Then run:
python scripts/validate_models.py --model "meta-llama/Llama-3.1-70B-Instruct"
```

#### Test Mixtral-8x22B (requires HuggingFace token):
```bash
# With token already set:
python scripts/validate_models.py --model "mistralai/Mixtral-8x22B-Instruct-v0.1"
```

---

### Option 3: Test with Smaller Models (For Validation Framework Testing)

If you just want to verify the validation script works without downloading large models:

```bash
# Test with GPT-2 (small, publicly available)
python scripts/validate_models.py --model "gpt2" --test-mode
```

This will:
- Load GPT-2 model (fast, ~57 seconds)
- Test loading, generation, memory tracking
- Generate validation report
- Verify the framework works correctly

---

## Authentication Setup (If Needed)

For models that require HuggingFace authentication:

### Step 1: Accept License
- Visit the model page on HuggingFace (e.g., https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct)
- Click "Agree and access repository"

### Step 2: Get Token
- Visit: https://huggingface.co/settings/tokens
- Create a new token with "Read" access
- Copy the token

### Step 3: Set Token (Windows PowerShell)
```powershell
$env:HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
```

### Step 4: Verify Token
```bash
python -c "import os; print('Token set:', bool(os.getenv('HUGGINGFACE_HUB_TOKEN')))"
```

---

## Expected Results

After running the validation, you'll get:

1. **JSON Results File**: `outputs/validation/models/model_validation_YYYYMMDD_HHMMSS.json`
   - Complete validation data
   - Loading times
   - Memory usage (before/after/delta)
   - Model information
   - Generation test results

2. **Markdown Summary**: `outputs/validation/models/MODEL_VALIDATION_SUMMARY.md`
   - Human-readable summary
   - System information
   - Model validation results
   - Loading times and memory usage documented

---

## Quick Test (Recommended First Step)

To verify everything works before testing large models:

```bash
# Test with GPT-2 (takes ~1 minute)
python scripts/validate_models.py --model "gpt2" --test-mode
```

This will:
- ✅ Verify validation script works
- ✅ Test model loading framework
- ✅ Generate documentation format
- ✅ Confirm memory tracking works

---

## Completion Checklist

After running the commands above, you should have:

- [x] Model support system implemented
- [x] Quantization support (4-bit/8-bit)
- [ ] Test results for at least one model (or all three if auth available)
- [ ] Documentation file with loading times and memory usage

**Note**: Even if some models fail due to authentication, the validation framework is complete and will document:
- Which models were tested
- Which succeeded/failed/skipped
- Loading times for successful loads
- Memory usage for successful loads
- Clear error messages for failures

---

## Troubleshooting

### Model Not Found Error
- Check the exact model name on HuggingFace
- Verify the model exists and is publicly available
- Some models may have been renamed or removed

### Authentication Required
- Set `HUGGINGFACE_HUB_TOKEN` environment variable
- Accept the license on HuggingFace
- Some models require special access

### Out of Memory
- Use smaller models for testing
- Enable quantization (already configured)
- Test on system with more GPU memory

### Slow Download
- First run downloads models (can take hours for large models)
- Subsequent runs use cached models (much faster)
- Consider using `hf_xet` package for faster downloads: `pip install hf_xet`

---

## Next Steps After Completion

Once Section 4.1 is complete:
1. Review the generated validation summary
2. Document any issues or special requirements
3. Proceed to Section 4.2 (Neuromodulation Packs Implementation)

