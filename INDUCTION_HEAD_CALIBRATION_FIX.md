# Induction Head Calibration Fragility Fix

## Problem

The induction head detection implementation was fragile because:

1. **Tokenization Sensitivity**: The calibration prompt "the cat sat the dog" assumes tokenization splits "the" the same way both times, but:
   - Leading spaces can create different token IDs (e.g., "the" vs "Ġthe" in GPT-2 style tokenizers)
   - Llama-3 tokenizers might handle "the" differently at different positions
   - Tokenizer prefixes (e.g., Ġ, ▁) can cause mismatches

2. **String Matching Assumption**: The code was relying on finding repeated tokens, but didn't account for tokenizer-specific prefixes that might cause the same word to be tokenized differently.

## Solution

### Changes Made

1. **More Robust Calibration Prompt**:
   - Changed from `"the cat sat the dog"` to `"A B C D E F A"`
   - Single letters are more likely to tokenize consistently across tokenizers
   - Clearer pattern: A appears at positions 0 and 6

2. **Token ID-Based Detection**:
   - Already using token IDs directly (not string matching)
   - Improved fallback logic to check for tokenizer variants
   - Added explicit handling for tokenizer prefixes

3. **Variant Handling**:
   - When no repeated tokens found, check variants: "A", " A", "A ", " A "
   - This accounts for tokenizer prefixes (e.g., ĠA vs A)
   - Uses token IDs directly for comparison

4. **Better Fallback**:
   - If variants don't work, check if first token appears multiple times
   - Added debug logging to help diagnose tokenization issues
   - Falls back to heuristic if all detection methods fail

### Code Changes

**Before**:
```python
calibration_prompt = "the cat sat the dog"
# ... tokenize and find repeated tokens
# Assumes "the" tokenizes the same way both times
```

**After**:
```python
calibration_prompt = "A B C D E F A"
# ... tokenize and find repeated tokens
# Check for tokenizer variants (A,  A, A ,  A )
# Use token IDs directly for comparison
# Handle tokenizer prefixes explicitly
```

### Why This Is Better

1. **Single Letters**: More consistent tokenization across tokenizers
2. **Clear Pattern**: A appears at positions 0 and 6, making detection straightforward
3. **Variant Handling**: Explicitly checks for tokenizer prefixes
4. **Token ID Direct**: Uses token IDs directly, not string matching
5. **Robust Fallback**: Multiple fallback strategies before giving up

### Safety Net

The `_heuristic_induction_heads()` fallback remains as a safety net:
- If detection fails, falls back to middle heads heuristic
- This ensures the effect still works even if calibration fails
- But real detection is preferred for scientific accuracy

### Files Modified

- `neuromod/effects.py`:
  - `QKScoreScalingEffect._detect_induction_heads()`:
    - Changed calibration prompt from "the cat sat the dog" to "A B C D E F A"
    - Improved token variant handling
    - Added debug logging
    - Better fallback logic

### Impact

- **Before**: Fragile to tokenizer differences, might fail on Llama-3 tokenizers
- **After**: More robust to tokenizer differences, handles prefixes explicitly
- **Safety**: Heuristic fallback ensures the effect still works if detection fails

### Testing Recommendations

1. Test with different tokenizers (GPT-2, Llama-3, Qwen)
2. Verify detection works with leading spaces
3. Check that fallback to heuristic works if detection fails
4. Ensure induction heads are detected correctly across architectures

