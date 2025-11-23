# LSD Class and Isomorphism Fixes

## Issue 2: Fix the "LSD" Class (Sample Size Negligence)

### Problem
Steering vectors were calculated from only N=5 prompts. In a 4096-dimensional space, this vector is just noise + specific token embeddings, not a robust signal.

### Solution: Robust Mean Difference Vector (MDV) Pipeline

#### 1. Dataset-Based Prompt Loading
- **Created**: `datasets/steering_prompts.jsonl` with 100+ prompt pairs
- **Format**: JSONL with `steering_type`, `positive`, and `negative` fields
- **Coverage**: 10 steering types × 10+ pairs each = 100+ total pairs

#### 2. Multi-Layer Activation Extraction
- **Before**: Only extracted from last layer (`layer_idx=-1`)
- **After**: Extracts from all layers and aggregates
- **Benefit**: Captures steering signal across the entire model depth

#### 3. PCA-Based Denoising
- **Before**: Simple mean difference vector (noisy in high dimensions)
- **After**: Compute PCA on difference vectors, use First Principal Component (PC1)
- **Algorithm**:
  1. Compute difference vectors: `diff = x_pos - x_neg` for each pair
  2. Stack all difference vectors: `[n_pairs, hidden_size]`
  3. Apply PCA: `PCA(n_components=1).fit(diff_vectors)`
  4. Extract PC1 as steering vector
- **Benefit**: Denoises the signal by capturing the dominant direction of variation

#### 4. Validation with Statistical Testing
- **Method**: Project validation prompts onto steering vector
- **Test**: t-test to check if positive/negative prompts are significantly separated
- **Threshold**: p < 0.01 (reject vectors that are just noise)
- **Benefit**: Ensures steering vectors actually work before using them

### Implementation

```python
def compute_vector_robust(self, dataset_path, steering_type, 
                        layer_idx=None, use_pca=True, validate=True):
    """
    Compute robust steering vector using MDV pipeline.
    
    1. Load 100+ prompt pairs from JSONL
    2. Extract activations from all layers
    3. Compute difference vectors (x_pos - x_neg)
    4. Apply PCA to extract PC1
    5. Validate separation significance (p < 0.01)
    """
    # Load pairs
    pos_prompts, neg_prompts = self.load_prompt_pairs(dataset_path, steering_type)
    
    # Extract from all layers
    for layer_idx in all_layers:
        pos_acts = [get_activations(p, layer_idx) for p in pos_prompts]
        neg_acts = [get_activations(n, layer_idx) for n in neg_prompts]
        diff_vectors = pos_acts - neg_acts  # [n_pairs, hidden_size]
    
    # Apply PCA
    pca = PCA(n_components=1)
    pca.fit(all_diff_vectors)
    steering_vec = pca.components_[0]  # PC1
    
    # Validate
    if validate:
        is_valid = self._validate_separation(steering_vec, val_pos, val_neg)
        if not is_valid:
            raise ValueError("Separation not significant (p >= 0.01)")
    
    return steering_vec
```

### Files Modified

- `neuromod/steering_generator.py`:
  - Added `load_prompt_pairs()`: Loads from JSONL dataset
  - Added `compute_vector_robust()`: New robust MDV pipeline
  - Added `_validate_separation()`: Statistical validation
  - Kept `compute_vector()` for backward compatibility (legacy method)

- `datasets/steering_prompts.jsonl` (new):
  - 100+ prompt pairs across 10 steering types
  - Format: `{"steering_type": "...", "positive": "...", "negative": "..."}`

### Benefits

1. **Robust Signal**: 100+ pairs vs. 5 pairs = 20x more data
2. **Denoised**: PCA extracts true signal direction, not noise
3. **Validated**: Statistical testing ensures vectors actually work
4. **Multi-Layer**: Captures steering across entire model depth
5. **Scalable**: Easy to add more prompt pairs to dataset

---

## Issue 3: Fix the "Isomorphism" Lies (Architecture Surgery vs. Logit Bias)

### Problem
`StructuredPrefacesEffect` and `LexicalJitterEffect` were implemented as `LogitsProcessor`, which is **causally incorrect** for "memory" or "input" perturbations.

**Why it's wrong:**
- LogitsProcessor modifies token probabilities **after** the model has already processed the input
- For "memory" perturbations (like prefaces), we need to modify the **input representation** or **KV cache**
- For "input" perturbations (like lexical jitter), we need to modify **embeddings**, not logits

### Solution: Move to Correct Causal Level

#### 1. LexicalJitterEffect → Embedding Layer

**Before (WRONG):**
```python
def get_logits_processor(self):
    # Modifies token scores AFTER processing
    # This is causally incorrect for "input" perturbations
    class LexicalJitterProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # Modify scores (too late!)
            scores = perturb_scores(scores)
            return scores
```

**After (CORRECT):**
```python
def apply(self, model, **kwargs):
    # Modify embeddings BEFORE processing
    # This is causally correct for "input" perturbations
    embedding_layer = model.get_input_embeddings()
    
    def jittered_forward(input_ids):
        embeddings = original_forward(input_ids)
        # Apply jitter to embeddings (correct causal structure)
        embeddings = add_noise(embeddings)  # or swap synonyms, etc.
        return embeddings
    
    embedding_layer.forward = jittered_forward
```

#### 2. StructuredPrefacesEffect → KV-Cache Level

**Before (WRONG):**
```python
def get_logits_processor(self):
    # Modifies token scores AFTER processing
    # This is causally incorrect for "memory" perturbations
    class StructuredPrefacesProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # Modify scores (too late!)
            scores = inject_preface_bias(scores)
            return scores
```

**After (CORRECT):**
```python
def apply(self, model, **kwargs):
    # Inject into KV cache (attention keys/values)
    # This is causally correct for "invisible" prefaces
    blocks = self._resolve_blocks(model)
    
    for block in blocks:
        attn_module = block.self_attn
        
        def kv_hook(module, input_tuple, output):
            # Modify K/V representations (correct causal structure)
            hidden_states = output[0]
            hidden_states = hidden_states + preface_bias
            return (hidden_states,) + output[1:]
        
        attn_module.k_proj.register_forward_hook(kv_hook)
        attn_module.v_proj.register_forward_hook(kv_hook)
```

### Implementation Details

#### LexicalJitterEffect
- **Location**: Embedding layer (`model.get_input_embeddings()`)
- **Method**: Hook `embedding_layer.forward()` to modify embeddings
- **Types**:
  - `synonym_swap`: Add noise to embeddings
  - `ablation`: Zero out fraction of embeddings
  - `noise_injection`: Add perceptual noise
  - `reframing`: Rotate embeddings slightly

#### StructuredPrefacesEffect
- **Location**: KV-cache (K and V projections in attention)
- **Method**: Hook `k_proj` and `v_proj` to inject bias
- **Effect**: "Invisible" preface affects attention without modifying input text
- **Causal Structure**: Correct - modifies memory/attention, not output logits

### Files Modified

- `neuromod/effects.py`:
  - `LexicalJitterEffect.apply()`: Now modifies embeddings (not logits)
  - `LexicalJitterEffect.get_logits_processor()`: Returns `None` (no longer uses logits)
  - `StructuredPrefacesEffect.apply()`: Now injects into KV-cache (not logits)
  - `StructuredPrefacesEffect.get_logits_processor()`: Returns `None` (no longer uses logits)

### Benefits

1. **Causally Correct**: Effects now modify the right part of the computation graph
2. **Proper Isomorphism**: "Memory" effects affect memory (KV-cache), "input" effects affect input (embeddings)
3. **Architectural Fidelity**: Matches how these effects would work in biological systems
4. **No Leakage**: Effects can't accidentally modify outputs in unintended ways

### Impact

- **LSD packs** (and other hallucinogen packs) that use steering vectors will now have robust, validated vectors
- **Memory perturbation effects** (like structured prefaces) will work correctly at the KV-cache level
- **Input perturbation effects** (like lexical jitter) will work correctly at the embedding level
- **No more isomorphism violations**: Effects are now at the correct causal level

