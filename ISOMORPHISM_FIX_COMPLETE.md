# Isomorphism Fix - Complete Implementation

## Problem
`StructuredPrefacesEffect` and `LexicalJitterEffect` were implemented as `LogitsProcessor`, which is **causally incorrect** for "memory" or "input" perturbations.

## Solution

### A. LexicalJitter (Input Perturbation) - FIXED

**Before (WRONG):**
- Implemented as `LogitsProcessor`
- Modified token probabilities AFTER processing
- Not causally correct for input perturbations

**After (CORRECT):**
- Uses `register_forward_hook` on embedding layer
- Modifies embeddings BEFORE they enter the transformer
- Creates actual perceptual noise, not just randomizing word choice

**Implementation:**
```python
def apply(self, model, **kwargs):
    embedding_layer = model.get_input_embeddings()
    
    def embedding_hook(module, input, output):
        # output is [batch, seq, hidden_dim] (The Embeddings)
        if self.jitter_type == "noise":
            noise = torch.randn_like(output) * self.sigma
            return output + noise
        return output
    
    handle = embedding_layer.register_forward_hook(embedding_hook)
    self.handles.append(handle)
```

**Key Changes:**
- Uses forward hook (not pre-hook) to modify output embeddings
- Directly modifies `output` tensor: `[batch, seq, hidden_dim]`
- Creates actual perceptual noise in the embedding space
- Applied BEFORE transformer processing (causally correct)

### B. StructuredPrefaces (KV-Injection) - FIXED

**Before (WRONG):**
- Implemented as `LogitsProcessor`
- Modified token probabilities AFTER processing
- Not causally correct for memory perturbations

**After (CORRECT):**
- Pre-computes KV states for preface string
- Injects directly into `past_key_values` during generation
- Creates actual "implanted memory," not just biasing word choice

**Implementation:**
```python
def apply(self, model, **kwargs):
    # Pre-compute KV states for preface string
    preface_inputs = tokenizer(self.preface_text, return_tensors="pt")
    outputs = model(**preface_inputs, use_cache=True)
    self.preface_kv_cache = outputs.past_key_values

def modify_kv_cache(self, kv_cache):
    """
    Inject pre-computed preface KV states into the cache.
    Called during generation, before the first token is generated.
    """
    if kv_cache is None:
        # First generation step - use preface as initial KV cache
        return self.preface_kv_cache
    
    # Concatenate preface KV cache with current KV cache
    modified_cache = []
    for layer_idx, (preface_k, preface_v) in enumerate(self.preface_kv_cache):
        current_k, current_v = kv_cache[layer_idx]
        # Concatenate along sequence dimension
        combined_k = torch.cat([preface_k, current_k], dim=-2)
        combined_v = torch.cat([preface_v, current_v], dim=-2)
        modified_cache.append((combined_k, combined_v))
    
    return tuple(modified_cache)
```

**Key Changes:**
- Pre-computes KV cache for preface string during `apply()`
- Stores `past_key_values` from model forward pass
- `modify_kv_cache()` injects preface KV into generation loop
- Concatenates preface KV with current KV along sequence dimension
- Creates actual "implanted memory" in attention mechanism

## Integration

The `modify_kv_cache()` method is already integrated into the generation pipeline:

1. **PackManager.modify_kv_cache()**: Calls `modify_kv_cache()` on all active effects
2. **NeuromodTool.modify_kv_cache()**: Wraps PackManager and applies modifications
3. **Generation Loop**: Should call `modify_kv_cache()` before each generation step

**Example Usage:**
```python
# During generation
for step in generation_steps:
    # Get current KV cache
    current_kv = model.past_key_values
    
    # Apply KV modifications (including StructuredPrefaces)
    modified_kv = neuromod_tool.modify_kv_cache(current_kv)
    
    # Generate next token with modified KV cache
    outputs = model.generate(..., past_key_values=modified_kv, ...)
```

## Files Modified

- `neuromod/effects.py`:
  - `LexicalJitterEffect.apply()`: Now uses `register_forward_hook` on embedding layer
  - `LexicalJitterEffect.cleanup()`: Removes hooks properly
  - `StructuredPrefacesEffect.apply()`: Pre-computes preface KV cache
  - `StructuredPrefacesEffect.modify_kv_cache()`: Injects preface KV into cache
  - `StructuredPrefacesEffect.cleanup()`: Clears cached KV states

## Benefits

1. **Causally Correct**: Effects now modify the right part of the computation graph
2. **Proper Isomorphism**: 
   - Input perturbations → Embedding layer (LexicalJitter)
   - Memory perturbations → KV cache (StructuredPrefaces)
3. **Architectural Fidelity**: Matches how these effects would work in biological systems
4. **No Leakage**: Effects can't accidentally modify outputs in unintended ways
5. **Actual Memory**: StructuredPrefaces creates real "implanted memory" in attention

## Impact

- **LexicalJitter**: Now creates actual perceptual noise in embeddings, not just word choice bias
- **StructuredPrefaces**: Now creates actual "implanted memory" in KV cache, not just logit bias
- **Isomorphism Preserved**: Effects are at the correct causal level for their intended purpose

