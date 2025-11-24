# Advanced Usage: Pack Optimization and Customization

This guide covers advanced topics including pack optimization, custom effect creation, and extending the framework.

## Pack Optimization

### Overview

The pack optimization system uses machine learning algorithms to automatically tune pack parameters to achieve specific behavioral targets.

### Behavioral Targets

Define what you want the pack to achieve:

```python
from neuromod.optimization.targets import TargetManager

target_manager = TargetManager()

# Create a custom target
target = target_manager.create_target(
    name="motivational_energetic",
    description="Increase motivation, energy, and productivity while maintaining focus"
)

# Add specific targets
target.add_emotion_target("joy", 0.7, weight=1.5)
target.add_emotion_target("excitement", 0.6, weight=1.0)
target.add_behavior_target("productivity", 0.8, weight=2.0)
target.add_behavior_target("focus", 0.7, weight=1.5)
target.add_metric_target("optimism", 0.6, weight=1.0)

# Add test prompts
target.test_prompts = [
    "What are your goals for today?",
    "How do you stay motivated?",
    "Describe your ideal work environment"
]
```

### Optimization Methods

Available methods:
- **Random Search**: Simple random parameter exploration
- **Evolutionary**: Genetic algorithm with mutation and crossover
- **Bayesian Optimization**: Gaussian process-based optimization
- **Reinforcement Learning**: RL-based parameter tuning

### Using the Drug Design Laboratory

```python
from neuromod.optimization.laboratory import DrugDesignLab
from neuromod.optimization.pack_optimizer import OptimizationMethod

# Create laboratory
lab = DrugDesignLab(model_name="meta-llama/Llama-3.1-8B-Instruct")

# Create a session
session = lab.create_session(
    target_name="joyful_social",
    base_pack_name="none"
)

# Optimize the pack
result = lab.optimize_pack(
    session_id=session.session_id,
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=100
)

# Test the optimized pack
test_result = lab.test_pack(session.session_id)
```

### Command-Line Interface

```bash
# List available targets
python -m neuromod.optimization.cli list-targets

# Test a pack against a target
python -m neuromod.optimization.cli test-pack \
    --target joyful_social \
    --prompts "Hello" "How are you?"

# Optimize a pack
python -m neuromod.optimization.cli optimize \
    --pack mdma \
    --target mdma_ecstasy \
    --method evolutionary
```

## Creating Custom Effects

### Base Effect Class

All effects inherit from `BaseEffect`:

```python
from neuromod.effects import BaseEffect
from transformers import LogitsProcessor

class CustomEffect(BaseEffect):
    """Custom neuromodulation effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", custom_param: float = 1.0):
        super().__init__(weight, direction)
        self.custom_param = custom_param
        self.handles = []  # Store hooks for cleanup
    
    def apply(self, model, **kwargs):
        """Apply the effect to the model"""
        # Example: Hook into attention mechanism
        def attention_hook(module, input_tuple, output):
            # Modify output
            modified_output = self._modify_attention(output)
            return modified_output
        
        # Register hook
        for layer in model.model.layers:
            handle = layer.self_attn.register_forward_hook(attention_hook)
            self.handles.append(handle)
    
    def _modify_attention(self, attention_output):
        """Modify attention output"""
        # Your custom logic here
        return attention_output
    
    def get_logits_processor(self) -> Optional[LogitsProcessor]:
        """Return logits processor if needed"""
        # Return None if not using logits processor
        return None
    
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
```

### Registering Custom Effects

```python
from neuromod.effects import EffectRegistry

# Register your custom effect
registry = EffectRegistry()
registry.register_effect("custom_effect", CustomEffect)

# Now you can use it in packs
```

## Creating Custom Packs

### Pack JSON Structure

```json
{
  "name": "custom_pack",
  "description": "Custom neuromodulation profile",
  "effects": [
    {
      "effect": "temperature",
      "weight": 0.5,
      "direction": "up",
      "parameters": {}
    },
    {
      "effect": "custom_effect",
      "weight": 0.7,
      "direction": "up",
      "parameters": {
        "custom_param": 1.5
      }
    }
  ]
}
```

### Adding to Pack Registry

```python
from neuromod.pack_system import PackRegistry

# Load your custom pack
registry = PackRegistry("packs/config.json")
pack = registry.load_pack("custom_pack.json")

# Or create programmatically
from neuromod.pack_system import Pack, EffectConfig

pack = Pack(
    name="custom_pack",
    description="Custom pack",
    effects=[
        EffectConfig(
            effect="temperature",
            weight=0.5,
            direction="up"
        )
    ]
)
```

## Extending the Framework

### Adding New Test Types

```python
from neuromod.testing.base_test import BaseTest

class CustomTest(BaseTest):
    """Custom psychometric test"""
    
    def __init__(self, model_name: str = "gpt2", test_mode: bool = True):
        super().__init__(model_name, test_mode)
        self.test_items = [
            "Item 1: ...",
            "Item 2: ..."
        ]
    
    def run_test(self, neuromod_tool=None) -> Dict[str, Any]:
        """Run the test"""
        results = []
        for item in self.test_items:
            response = self.generate_response_safe(item)
            score = self._score_response(response)
            results.append(score)
        
        return {
            "test_name": "custom_test",
            "scores": results,
            "total_score": sum(results)
        }
    
    def _score_response(self, response: str) -> float:
        """Score a response"""
        # Your scoring logic
        return 0.0
```

### Custom Telemetry Metrics

```python
from neuromod.testing.telemetry import TelemetryCollector

class CustomTelemetryCollector(TelemetryCollector):
    """Custom telemetry metrics"""
    
    def collect_custom_metric(self, responses: List[str]) -> float:
        """Collect a custom metric"""
        # Your metric calculation
        return 0.0
```

## Integration Examples

### Using with Custom Models

```python
from neuromod.model_support import ModelSupportManager
from neuromod.neuromod_tool import NeuromodTool
from neuromod.pack_system import PackRegistry

# Load your custom model
model_manager = ModelSupportManager(test_mode=False)
model, tokenizer, model_info = model_manager.load_model("your-model-name")

# Create neuromod tool
pack_registry = PackRegistry("packs/config.json")
neuromod_tool = NeuromodTool(
    registry=pack_registry,
    model=model,
    tokenizer=tokenizer
)

# Apply a pack
neuromod_tool.apply(pack_name="caffeine", intensity=0.7)

# Generate with neuromodulation
response = model.generate(input_ids, max_new_tokens=100)
```

### Batch Processing

```python
from neuromod.testing.endpoint_calculator import EndpointCalculator

calculator = EndpointCalculator()

# Process multiple packs
packs = ["caffeine", "lsd", "alcohol"]
results = {}

for pack in packs:
    result = calculator.calculate_endpoints(
        pack_name=pack,
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    )
    results[pack] = result
```

## Best Practices

### Effect Design
- **Keep effects focused**: Each effect should do one thing well
- **Use appropriate hooks**: Embedding layer for input effects, attention for attention effects
- **Clean up resources**: Always implement `cleanup()` to remove hooks

### Pack Design
- **Start simple**: Begin with 1-2 effects, add complexity gradually
- **Test incrementally**: Test each effect individually before combining
- **Document parameters**: Include clear descriptions of what each parameter does

### Optimization
- **Define clear targets**: Specific, measurable behavioral goals
- **Use appropriate test prompts**: Prompts that will reveal the target behavior
- **Validate results**: Check that optimized packs actually achieve targets

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues.

## Vertex AI Deployment

This section covers deploying the neuromodulation system with Google Cloud Vertex AI for pay-per-use model serving with Llama 3 and other large models.

### Why Vertex AI for Pay-Per-Use?

**Benefits**:
- ✅ Pay only for predictions (not idle time)
- ✅ Auto-scaling to zero when not in use
- ✅ GPU acceleration for large models
- ✅ Managed infrastructure
- ✅ Integration with Google Cloud ecosystem
- ✅ Llama 3.1 70B support with A100 GPUs

**Cost Structure**:
- Input processing: $0.0025 per 1000 characters
- Output generation: $0.01 per 1000 characters
- Model hosting: $0.0001 per 1000 characters
- Example for 1000 tokens (~750 characters): ~$0.00945 per 1000 tokens

### Quick Start

**Step 1: Setup Google Cloud Project**
```bash
# Install Google Cloud SDK
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

**Step 2: Deploy Vertex AI Endpoint**
```bash
# Make deployment script executable
chmod +x api/deploy_vertex_ai.sh

# Deploy Llama 3.1 8B (recommended for testing)
./api/deploy_vertex_ai.sh \
  --project-id YOUR_PROJECT_ID \
  --model-name meta-llama/Meta-Llama-3.1-8B \
  --endpoint-name neuromod-llama-8b \
  deploy

# Deploy Llama 3.1 70B (for production)
./api/deploy_vertex_ai.sh \
  --project-id YOUR_PROJECT_ID \
  --model-name meta-llama/Meta-Llama-3.1-70B \
  --endpoint-name neuromod-llama-70b \
  deploy
```

**Step 3: Deploy Cloud Run Frontend**
```bash
# Deploy the API server
gcloud run deploy neuromodulation-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 10 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

**Step 4: Test the System**
```bash
# Get the Cloud Run URL
CLOUD_RUN_URL=$(gcloud run services describe neuromodulation-api --region=us-central1 --format="value(status.url)")

# Test with caffeine pack
curl -X POST "$CLOUD_RUN_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me about coffee",
    "pack_name": "caffeine",
    "max_tokens": 100
  }'
```

### Model Compatibility

**Supported Models**:
- **Llama 3 Series**: meta-llama/Meta-Llama-3.1-8B (T4 GPU), meta-llama/Meta-Llama-3.1-70B (A100 GPU)
- **Qwen Series**: Qwen/Qwen2.5-7B (T4 GPU), Qwen/Qwen2.5-32B (A100 GPU)
- **Mixtral Series**: mistralai/Mixtral-8x7B-v0.1 (V100 GPU)

**GPU Requirements**:
- **T4 GPU (16GB VRAM)**: Llama 3.1 8B (4-bit quantized), Qwen 2.5 7B (4-bit quantized), Lower cost
- **A100 GPU (40GB VRAM)**: Llama 3.1 70B (4-bit quantized), Qwen 2.5 32B (4-bit quantized), Higher cost, better quality

### API Integration

**Model Loading**:
```bash
# Load local model (for small models)
POST /model/load?model_name=microsoft/DialoGPT-medium

# Load Vertex AI model (for large models)
POST /model/load?model_name=meta-llama/Meta-Llama-3.1-70B
```

**Text Generation**:
```bash
# Generate with neuromodulation effects
POST /generate
{
  "prompt": "Tell me about coffee",
  "pack_name": "caffeine",
  "max_tokens": 100,
  "temperature": 1.0,
  "top_p": 1.0
}
```

**Response Format**:
```json
{
  "generated_text": "Coffee is a stimulating beverage...",
  "pack_applied": "caffeine",
  "generation_time": 2.5,
  "model_type": "vertex_ai"
}
```

### Cost Optimization

**Development Setup**:
- Llama 3.1 8B on T4 GPU: ~$0.005 per 1000 tokens, ~$15/month for 1000 requests/day

**Production Setup**:
- Llama 3.1 70B on A100 GPU: ~$0.015 per 1000 tokens, ~$45/month for 1000 requests/day

### Troubleshooting

**Endpoint Creation Fails**:
```bash
# Check permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID

# Ensure APIs are enabled
gcloud services list --enabled | grep aiplatform
```

**Container Build Fails**:
```bash
# Check Docker installation
docker --version

# Authenticate with Container Registry
gcloud auth configure-docker
```

**Model Loading Fails**:
```bash
# Check GPU availability
nvidia-smi

# Verify model name
curl "https://huggingface.co/api/models/meta-llama/Meta-Llama-3.1-8B"
```

**High Costs**:
```bash
# Monitor usage
gcloud ai endpoints list --region=us-central1

# Set up billing alerts
# https://console.cloud.google.com/billing/alerts
```

### Security Considerations

**Authentication**:
```bash
# Use service accounts for production
gcloud iam service-accounts create neuromod-service

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:neuromod-service@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

**Network Security**:
```bash
# Use VPC for private endpoints
gcloud compute networks create neuromod-vpc

# Configure firewall rules
gcloud compute firewall-rules create allow-neuromod \
  --network neuromod-vpc \
  --allow tcp:8080
```

### Best Practices

1. **Model Selection**: Development: Use Llama 3.1 8B, Production: Use Llama 3.1 70B, Cost-sensitive: Use Qwen 2.5 7B
2. **Pack Management**: Pre-load common packs, Cache pack configurations, Monitor pack application success rates
3. **Cost Monitoring**: Set up billing alerts, Monitor prediction costs, Use cost optimization features
4. **Performance Optimization**: Use 4-bit quantization, Optimize batch sizes, Monitor GPU utilization

For more details, see the [Vertex AI Deployment Guide](vertex_container/README.md) in the `vertex_container/` directory.

## Examples

See `demo/pack_optimization_demo.py` for complete examples of:
- Creating custom targets
- Using the drug design laboratory
- Running optimization
- Testing optimized packs

