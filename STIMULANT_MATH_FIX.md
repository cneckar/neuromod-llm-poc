# Stimulant Math Fix: Proper Attention Sharpening

## Problem
The `QKScoreScalingEffect` (and `AttentionFocusEffect`) were scaling attention weights **after** the Softmax operation. This is mathematically ineffective because:

- If the attention distribution is `[0.9, 0.1]`, scaling it to `[0.99, 0.11]` and re-normalizing changes almost nothing
- The effect is minimal because softmax already normalizes the distribution
- This doesn't actually sharpen the attention focus

## Solution

### Mathematical Basis
Scaling the Query (Q) vectors **before** the dot product is mathematically equivalent to scaling the attention logits, which effectively lowers the temperature of the Softmax:

- **Original**: `attn_logits = Q @ K^T / sqrt(d_k)`
- **Scaled**: `attn_logits = (α * Q) @ K^T / sqrt(d_k) = α * (Q @ K^T) / sqrt(d_k)`
- **Result**: All logits are scaled by α, which sharpens the softmax distribution

### Implementation

#### `QKScoreScalingEffect` Changes:
1. **Removed post-softmax scaling**: No longer scales attention weights after softmax
2. **Added Q vector scaling**: Hooks into the Q projection layer and scales Q before dot product
3. **Architecture support**:
   - **Llama/Mistral/Qwen**: Hooks `block.self_attn.q_proj`
   - **GPT2**: Handles fused `block.attn.c_attn` by splitting Q, K, V, scaling only Q, then concatenating
   - **GPT-NeoX**: Handles `block.attention.query_key_value` similarly

#### `AttentionFocusEffect` Changes:
- Same fix applied (it had the same problem)
- Slightly more conservative scaling (max 1.3x vs 2.0x)

### Code Structure

```python
# Calculate scaling factor
# weight=0.0 -> scale=1.0 (baseline, no effect)
# weight=0.5 -> scale=1.5 (moderate sharpening)
# weight=1.0 -> scale=2.0 (maximum sharpening)
base_scale = 1.0
max_scale = 2.0
scale_factor = self.get_effective_value(base_scale, max_scale)

# Hook into Q projection layer
def q_hook(module, input_tuple, output):
    # For separate q_proj: output is Q tensor [batch, seq, head_dim]
    # For fused c_attn: output is [batch, seq, 3*head_dim] (Q, K, V)
    if fused:
        # Split, scale Q, concatenate
        Q, K, V = split_fused_output(output)
        Q_scaled = Q * scale_factor
        return concat(Q_scaled, K, V)
    else:
        # Scale Q directly
        return output * scale_factor

handle = q_proj.register_forward_hook(q_hook)
```

## Key Improvements

1. **Mathematically Sound**: Scales Q before dot product, not weights after softmax
2. **Effective Sharpening**: Actually changes the attention distribution significantly
3. **Architecture Agnostic**: Handles Llama, GPT2, GPT-NeoX, and other architectures
4. **Proper Hooking**: Uses forward hooks on Q projection layer
5. **Clean Cleanup**: Properly removes hooks when effect is cleaned up

## Testing

To verify the fix works:

1. **Before fix**: Attention weights after softmax scaling had minimal effect
2. **After fix**: Q scaling before dot product should produce noticeable sharpening

You can test by:
- Applying a stimulant pack (caffeine, amphetamine, etc.) that uses `qk_score_scaling`
- Observing that attention becomes more focused/sharp
- Comparing with baseline to see the difference

## Files Modified

- `neuromod/effects.py`:
  - `QKScoreScalingEffect`: Complete rewrite to scale Q before dot product
  - `AttentionFocusEffect`: Same fix applied

## Impact

This fix makes stimulant effects (caffeine, amphetamine, methylphenidate, etc.) that use `qk_score_scaling` actually work as intended. The attention sharpening will now be effective rather than being mathematically nullified by the softmax normalization.

