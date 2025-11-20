# Model Loading Documentation

## Overview

This document provides documentation for model loading times and memory usage for the three primary models used in the neuromodulation study.

## Primary Models

1. **meta-llama/Llama-3.1-70B-Instruct**
   - Size: 70B parameters
   - Quantization: 4-bit (required for memory efficiency)
   - Authentication: Required (HuggingFace token)
   - Expected VRAM: ~40GB with 4-bit quantization
   - Expected Loading Time: 5-15 minutes (depending on download speed)

2. **Qwen/Qwen-2.5-Omni-7B**
   - Size: 7B parameters
   - Quantization: 8-bit (default)
   - Authentication: Not required
   - Expected VRAM: ~8-12GB with 8-bit quantization
   - Expected Loading Time: 2-5 minutes

3. **mistralai/Mixtral-8x22B-Instruct-v0.1**
   - Size: 8×22B parameters (Mixture of Experts)
   - Quantization: 4-bit (required for memory efficiency)
   - Authentication: Required (HuggingFace token)
   - Expected VRAM: ~45-50GB with 4-bit quantization
   - Expected Loading Time: 10-20 minutes

## Validation Script

A validation script has been created at `scripts/validate_models.py` to test model loading and document metrics.

### Usage

```bash
# Test all three primary models
python scripts/validate_models.py

# Test a specific model
python scripts/validate_models.py --model "Qwen/Qwen-2.5-Omni-7B"

# Specify output directory
python scripts/validate_models.py --output-dir outputs/validation/models
```

### What It Tests

1. **Model Loading**: Attempts to load each model with proper quantization
2. **Loading Time**: Measures time from start to successful load
3. **Memory Usage**: Tracks GPU and system RAM usage before/after loading
4. **Generation Test**: Performs a simple generation to verify model works
5. **Quantization Status**: Verifies quantization is applied correctly

### Output

The script generates:
- `model_validation_YYYYMMDD_HHMMSS.json`: Complete validation results
- `MODEL_VALIDATION_SUMMARY.md`: Human-readable summary

## Prerequisites

1. **HuggingFace Authentication** (for Meta Llama and Mixtral models):
   ```bash
   # Set token as environment variable
   export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
   
   # Or use CLI login
   huggingface-cli login
   ```

2. **Required Packages**:
   ```bash
   pip install torch transformers accelerate bitsandbytes psutil
   ```

3. **GPU Requirements**:
   - Minimum: 20GB VRAM for 70B models (with 4-bit quantization)
   - Recommended: 40GB+ VRAM for comfortable operation
   - For 7B models: 8-12GB VRAM sufficient

## Expected Results

### Successful Load Indicators

- ✅ Model loads without errors
- ✅ Loading time is reasonable (< 20 minutes)
- ✅ Memory usage is within expected ranges
- ✅ Generation test produces valid output
- ✅ Quantization is correctly applied (if specified)

### Common Issues

1. **Authentication Errors**:
   - **Symptom**: "401 Unauthorized" or "model not found"
   - **Solution**: Set `HUGGINGFACE_HUB_TOKEN` environment variable
   - **For Meta Llama**: Accept license at https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

2. **Out of Memory**:
   - **Symptom**: CUDA out of memory error
   - **Solution**: Ensure 4-bit quantization is enabled for large models
   - **Alternative**: Use smaller model or reduce batch size

3. **Slow Loading**:
   - **Symptom**: Loading takes > 30 minutes
   - **Solution**: Check internet connection (first load downloads model)
   - **Note**: Subsequent loads should be faster (cached)

## Memory Usage Guidelines

| Model | Quantization | Expected VRAM | System RAM |
|-------|-------------|---------------|------------|
| Llama-3.1-70B | 4-bit | ~40GB | ~10GB |
| Qwen-2.5-Omni-7B | 8-bit | ~8-12GB | ~5GB |
| Mixtral-8×22B | 4-bit | ~45-50GB | ~15GB |

## Loading Time Benchmarks

Loading times vary based on:
- First load vs cached (first load includes download)
- Internet connection speed
- Disk I/O speed
- GPU initialization time

**Typical Loading Times** (cached models):
- 7B models: 2-5 minutes
- 70B models: 5-15 minutes
- MoE models: 10-20 minutes

## Next Steps

After validation:
1. Review `MODEL_VALIDATION_SUMMARY.md` for results
2. Verify all models load successfully
3. Document any issues or deviations from expected metrics
4. Proceed with Phase 1 (Pilot Study) once all models validated

## Notes

- Models are loaded in production mode (not test mode) for accurate metrics
- First run may take longer due to model download
- Ensure sufficient disk space (~100GB+ for all models)
- GPU memory is measured at peak usage during loading

