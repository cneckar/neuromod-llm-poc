# Troubleshooting Guide

Common issues and solutions for the Neuromod-LLM framework.

## Setup Issues

### Hugging Face Authentication

**Problem**: Models fail to load with authentication errors.

**Solution**:
```bash
# Quick setup (recommended)
python setup_hf_credentials.py

# Or use Hugging Face CLI
huggingface-cli login
```

**Get your token**: https://huggingface.co/settings/tokens

**Detailed guide**: See the [Hugging Face Credentials Guide](../HUGGINGFACE_CREDENTIALS_GUIDE.md) in the root directory.

### Missing Dependencies

**Problem**: Import errors for scikit-learn, statsmodels, etc.

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# Verify scikit-learn is installed
python -c "import sklearn; print(sklearn.__version__)"
```

**Note**: `scikit-learn>=1.3.0` is required (already in requirements.txt).

### Model Loading Failures

**Problem**: Models fail to load due to memory issues.

**Solutions**:
1. **Use quantization** (automatic for large models):
   - 4-bit quantization for 70B models
   - 8-bit quantization for 7B models

2. **Use smaller models for testing**:
   ```bash
   python scripts/validate_models.py --model gpt2 --test-mode
   ```

3. **Check available memory**:
   ```python
   import psutil
   print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
   ```

## Runtime Issues

### Out of Memory (OOM) Errors

**Problem**: GPU runs out of memory during model loading or generation.

**Solutions**:
1. **Use smaller models**: Switch to 7B or 8B models instead of 70B
2. **Enable CPU offloading**: Some models support CPU offloading
3. **Reduce batch size**: Process fewer items at once
4. **Clear cache**: `torch.cuda.empty_cache()`

### Slow Generation

**Problem**: Model generation is very slow.

**Solutions**:
1. **Use GPU**: Ensure CUDA is available: `torch.cuda.is_available()`
2. **Reduce max_new_tokens**: Generate shorter responses
3. **Use smaller models**: 7B models are much faster than 70B
4. **Check quantization**: Quantized models are faster

### Pack Application Errors

**Problem**: Packs fail to apply or cause errors.

**Solutions**:
1. **Validate packs first**:
   ```bash
   python scripts/validate_packs.py
   ```

2. **Check effect compatibility**: Some effects may not work with all model architectures

3. **Verify model support**: Ensure the model supports the required hooks

### Steering Vector Not Found

**Problem**: Error about missing steering vectors.

**Solution**:
```bash
# Generate steering vectors
python scripts/generate_steering_vectors.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --output-dir outputs/steering_vectors
```

**Note**: Steering vectors are generated using PCA on difference vectors from `datasets/steering_prompts.jsonl`.

## Data Collection Issues

### Endpoint Calculation Failures

**Problem**: `calculate_endpoints.py` fails or produces errors.

**Solutions**:
1. **Check model availability**: Ensure model can load
2. **Use --skip-completed**: Resume interrupted runs
3. **Check test mode**: Use `--test-mode` for test models
4. **Review logs**: Check error messages for specific issues

### Statistical Analysis Errors

**Problem**: Statistical analysis fails or produces incorrect results.

**Solutions**:
1. **Check data format**: Ensure endpoint files are valid JSON
2. **Verify sample sizes**: Need n≥126 per condition for power
3. **Check dependencies**: Ensure statsmodels, scipy are installed
4. **Review FDR correction**: Verify p-values are being corrected correctly

### Missing Data

**Problem**: Some tests fail to produce results.

**Solutions**:
1. **Check token limits**: Some tests need sufficient token budget
2. **Review test prompts**: Ensure prompts are valid
3. **Check model responses**: Some models may refuse certain prompts
4. **Use --verbose**: Get detailed error messages

## Experimental Design Issues

### Blinding Failures

**Problem**: Blinding audit finds leakage issues.

**Solutions**:
1. **Review prompts**: Ensure all prompts are generic
2. **Check pack descriptions**: Remove any pack-specific language
3. **Re-run audit**: `python scripts/audit_blinding.py`

### Latin Square Errors

**Problem**: Latin square generation fails or is unbalanced.

**Solutions**:
1. **Check prompt count**: Need at least 3 prompts for 3 conditions
2. **Verify randomization seed**: Same seed should produce same square
3. **Review experimental design**: Ensure conditions are properly assigned

## Performance Issues

### Slow Test Execution

**Problem**: Tests take too long to run.

**Solutions**:
1. **Reduce sample size**: Use smaller n for testing
2. **Use test models**: gpt2 is much faster than Llama-70B
3. **Parallelize**: Run multiple packs in parallel if possible
4. **Optimize token limits**: Reduce max_new_tokens where appropriate

### Memory Leaks

**Problem**: Memory usage increases over time.

**Solutions**:
1. **Clean up effects**: Ensure `cleanup()` is called
2. **Clear model cache**: `torch.cuda.empty_cache()` between runs
3. **Unload models**: Don't keep models loaded unnecessarily
4. **Use context managers**: Properly manage resources

## Platform-Specific Issues

### Windows Issues

**Problem**: Encoding errors, path issues, etc.

**Solutions**:
1. **UTF-8 encoding**: Files are read with UTF-8 encoding
2. **Path separators**: Use forward slashes or `pathlib.Path`
3. **PowerShell**: Use `;` instead of `&&` for command chaining

### Linux/Mac Issues

**Problem**: Permission errors, path issues.

**Solutions**:
1. **Check permissions**: Ensure write access to output directories
2. **Virtual environment**: Use venv or conda
3. **Python version**: Requires Python 3.8+

## Getting Help

### Check Logs

Most scripts produce detailed logs:
- Check console output for error messages
- Review `outputs/validation/` for validation results
- Check `outputs/endpoints/` for endpoint calculation results

### Common Error Messages

**"Model not found"**:
- Check model name spelling
- Verify Hugging Face authentication
- Ensure model is available on HuggingFace Hub

**"Pack not found"**:
- Check pack name in `packs/config.json`
- Verify pack file exists
- Check JSON syntax

**"Effect not found"**:
- Verify effect is registered in `EffectRegistry`
- Check effect name spelling
- Ensure effect class exists

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use `--verbose` flag in scripts:
```bash
python scripts/calculate_endpoints.py --pack caffeine --verbose
```

## Still Having Issues?

1. **Check GitHub Issues**: Search for similar problems
2. **Review Documentation**: See other guides in `docs/`
3. **Open an Issue**: Provide error messages, logs, and reproduction steps

