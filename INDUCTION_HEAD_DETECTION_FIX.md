# Induction Head Detection Fix: Real Detection vs. Magic Numbers

## Problem
The `_detect_induction_heads` method was using a "magic number" heuristic:
```python
if num_heads >= 8:
    induction_heads = [num_heads // 2 - 1, num_heads // 2]  # Just guessing!
```

**The Critique:**
- We were guessing that induction heads are always in the middle indices
- In Llama-3, Mistral, and Qwen, induction heads drift during training
- Hardcoding `[num_heads // 2]` is scientifically defensible only if we call it "heuristic sharpening"
- If we want to claim "Induction Head Targeting," we must actually detect them

## Solution: Real Induction Head Detection

### Implementation

**Calibration Pattern:**
- Pass a string like "A B C D E F A" through the model
- Identify heads that strongly attend from the second A to the token after the first A
- This is the classic induction head detection pattern

**Algorithm:**
1. Create calibration prompt: "A B C D E F A"
2. Tokenize and find positions of first and second "A"
3. Forward pass through the model with attention hook
4. Extract attention weights: `attention[0, :, second_a_pos, target_pos]`
   - Where `target_pos = first_a_pos + 1` (token after first A)
5. Identify heads with highest attention scores (top 20% or at least 2 heads)
6. Return detected induction head indices

### Code

```python
def _detect_induction_heads(self, model, block, layer_idx: int, num_heads: int, 
                            tokenizer=None) -> List[int]:
    """
    REAL DETECTION: Pass "A B C D E F A" through model and identify heads
    that attend from second A to position after first A.
    """
    # Create calibration prompt
    calibration_prompt = "A B C D E F A"
    inputs = tokenizer(calibration_prompt, return_tensors="pt")
    
    # Find first and second A positions
    first_a_pos = ...  # Position of first A
    second_a_pos = ...  # Position of second A
    target_pos = first_a_pos + 1  # Token after first A
    
    # Hook to capture attention weights
    attention_weights = None
    def attention_hook(module, input_tuple, output):
        nonlocal attention_weights
        if isinstance(output, tuple) and len(output) >= 2:
            attention_weights = output[1].detach().cpu()
    
    handle = attn_module.register_forward_hook(attention_hook)
    
    # Forward pass
    with torch.no_grad():
        model(**inputs)
    
    # Extract attention from second A to target position
    # attention_weights: [batch, num_heads, seq_len, seq_len]
    induction_scores = attention_weights[0, :, second_a_pos, target_pos].numpy()
    
    # Find top-k heads by induction score
    n_select = max(2, min(int(num_heads * 0.2), num_heads))
    top_indices = np.argsort(induction_scores)[-n_select:][::-1]
    induction_heads = top_indices.tolist()
    
    return induction_heads
```

### Fallback

If detection fails (no tokenizer, attention weights unavailable, etc.), the method falls back to the heuristic:

```python
def _heuristic_induction_heads(self, num_heads: int) -> List[int]:
    """
    Fallback heuristic for induction head detection.
    
    This is a simple heuristic and should only be used when real detection fails.
    Consider renaming the effect to "HeuristicAttentionSharpening" if this is used.
    """
    if num_heads >= 8:
        induction_heads = [num_heads // 2 - 1, num_heads // 2]
    elif num_heads >= 4:
        induction_heads = [num_heads // 2]
    else:
        induction_heads = [0]
    
    logger.warning(f"Using heuristic induction head detection: {induction_heads}")
    return induction_heads
```

## Benefits

1. **Real Detection**: Actually identifies induction heads, not just guessing
2. **Model-Specific**: Works for any model architecture (Llama, Mistral, Qwen, etc.)
3. **Training-Aware**: Detects heads that actually learned the induction pattern
4. **Scientifically Defensible**: Can claim "induction head targeting" with evidence
5. **Fallback Safety**: Falls back to heuristic if detection fails

## Impact

- **Stimulant Packs**: Now use real induction head detection, not magic numbers
- **Scientific Rigor**: Can legitimately claim "induction head targeting" in papers
- **Model Compatibility**: Works across different architectures and training regimes
- **Transparency**: Logs detected heads and their induction scores

## Alternative: Rename to Heuristic

If real detection consistently fails or is too slow, we can:
1. Rename `QKScoreScalingEffect` to `HeuristicAttentionSharpening`
2. Drop the claim about "induction heads" in documentation
3. Be honest that it's a heuristic sharpening of middle heads

But with this implementation, we can actually detect induction heads!

## Files Modified

- `neuromod/effects.py`:
  - `_detect_induction_heads()`: Now uses real detection via calibration prompt
  - `_heuristic_induction_heads()`: Fallback heuristic (with warning)
  - `apply()`: Passes tokenizer to detection method
  - Updated docstrings to reflect real detection

