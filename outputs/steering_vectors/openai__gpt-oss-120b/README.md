# Steering vectors — `openai/gpt-oss-120b`

Pre-computed Contrastive Activation Addition (CAA) steering vectors for **`openai/gpt-oss-120b`**,
committed so anyone running that model gets real steering **without regenerating** (regeneration
needs an H200-class GPU to load the MXFP4 checkpoint).

| Property | Value |
|---|---|
| Model | `openai/gpt-oss-120b` |
| Vector dim | **2880** (model `hidden_size`) |
| Layer | `-1` (last transformer block; files named `<type>_layer-1.pt`) |
| Norm | unit (‖v‖ = 1.0) |
| dtype | float32 (cast to the model's dtype at apply time) |
| Method | robust MDV + PCA (PC1 of the positive−negative activation differences), `scripts/generate_steering_vectors.py` |
| Types | abstract, affiliative, associative, creative, ego_thin, goal_focused, playful, prosocial, synesthesia, visionary |

## How they're loaded

The loader is model-aware (`neuromod.effects.resolve_steering_vector_path`): it looks under
`<vector_dir>/<model_slug>/` first, where `model_slug = "openai/gpt-oss-120b".replace("/", "__")`
= `openai__gpt-oss-120b` (this directory). The model id comes from the loaded model (local usage)
or the `MODEL_NAME` env (served worker), so no extra config is needed — running gpt-oss-120b picks
these up automatically, while other models fall back to their own subdir or the flat legacy set.

## Regenerating (if ever needed)

```bash
python scripts/generate_steering_vectors.py \
  --model openai/gpt-oss-120b \
  --output-dir outputs/steering_vectors \
  --layer -1 --no-validate
# writes back into outputs/steering_vectors/openai__gpt-oss-120b/
```

On the RunPod serverless deployment this is the `steering` task (runs on the warm H200 worker,
writes to the network volume). See `deploy/runpod/README.md`.
