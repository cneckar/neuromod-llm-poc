# Steering Vector Fix: Implementing Real CAA

## Problem
The `SteeringEffect` class in `neuromod/effects.py` was initializing steering vectors with `torch.randn()`, which is not scientifically sound. Random vectors don't represent actual contrastive activation differences.

## Solution

### Step A: Created `neuromod/steering_generator.py`
A new module that implements Contrastive Activation Addition (CAA) to generate steering vectors from actual model activations:

- **`SteeringVectorGenerator`** class that:
  - Extracts residual stream activations from transformer layers
  - Computes mean difference between positive and negative prompt activations
  - Normalizes vectors for stability
  - Handles multiple model architectures (Llama, GPT2, GPT-NeoX, etc.)

- **Pre-defined contrastive prompt pairs** for all steering types:
  - `associative`, `creative`, `visionary`, `synesthesia`, `prosocial`, `playful`, `ego_thin`, `goal_focused`, `abstract`, `affiliative`

### Step B: Updated `neuromod/effects.py`

#### `SteeringEffect` class changes:
1. **Removed random initialization**: No more `torch.randn()` calls
2. **Added `load_vector()` method**: Loads pre-computed vectors from disk
3. **Zero vector fallback**: Uses `torch.zeros()` instead of random noise when vectors aren't found
4. **Dynamic hidden size handling**: Automatically adjusts vector size to match model
5. **Vector caching**: Caches loaded vectors to avoid repeated disk I/O

#### `ActivationAdditionsEffect` class changes:
1. **Removed random initialization**: Replaced with empty dictionary
2. **Added `_load_vector()` method**: Same loading mechanism as `SteeringEffect`
3. **Updated `compute_contrastive_steering_vector()`**: Now uses disk loading instead of random fallback
4. **Deprecated `_extract_activations()`**: Returns zero vector with warning (use `SteeringVectorGenerator` instead)

### Step C: Created `scripts/generate_steering_vectors.py`
A utility script to generate all steering vectors:

```bash
# Generate all vectors for a model
python scripts/generate_steering_vectors.py --model gpt2 --test-mode

# Generate specific vector type
python scripts/generate_steering_vectors.py --model meta-llama/Llama-3.1-8B-Instruct --steering-type associative

# Generate for specific layer
python scripts/generate_steering_vectors.py --model gpt2 --layer -1 --test-mode
```

## Usage

### 1. Generate Steering Vectors
First, generate the vectors for your model(s):

```bash
# For test models
python scripts/generate_steering_vectors.py --model gpt2 --test-mode

# For production models
python scripts/generate_steering_vectors.py --model meta-llama/Llama-3.1-8B-Instruct
```

This will create files in `outputs/steering_vectors/` like:
- `associative_layer-1.pt`
- `creative_layer-1.pt`
- `visionary_layer-1.pt`
- etc.

### 2. Use in Packs
The `SteeringEffect` will automatically load vectors from `outputs/steering_vectors/` when used in packs. No changes needed to pack JSON files.

### 3. Custom Vector Paths
You can specify custom vector paths when creating effects:

```python
effect = SteeringEffect(
    weight=0.5,
    direction="up",
    steering_type="associative",
    vector_path="path/to/custom/vector.pt"
)
```

## File Structure

```
neuromod/
├── steering_generator.py      # NEW: CAA implementation
└── effects.py                  # UPDATED: Removed random vectors

scripts/
└── generate_steering_vectors.py  # NEW: Vector generation script

outputs/
└── steering_vectors/              # Generated vectors stored here
    ├── associative_layer-1.pt
    ├── creative_layer-1.pt
    └── ...
```

## Key Improvements

1. **Scientifically Sound**: Vectors are now computed from actual model activations using CAA
2. **No Random Noise**: Zero vectors used as failsafe instead of random noise
3. **Architecture Agnostic**: Works with Llama, GPT2, GPT-NeoX, and other architectures
4. **Automatic Loading**: Vectors are loaded automatically when effects are applied
5. **Caching**: Loaded vectors are cached to improve performance
6. **Flexible**: Supports custom vector paths and automatic size adjustment

## Migration Notes

- **Existing packs**: No changes needed, vectors will be loaded automatically
- **Missing vectors**: System will use zero vectors (no effect) instead of random noise
- **Warnings**: You'll see warnings if vectors aren't found - generate them using the script

## Next Steps

1. Generate vectors for all models you plan to use
2. Verify vectors are being loaded correctly (check logs)
3. Test packs to ensure steering effects work as expected
4. Consider generating vectors for multiple layers if needed

