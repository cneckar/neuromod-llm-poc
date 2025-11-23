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

## Examples

See `demo/pack_optimization_demo.py` for complete examples of:
- Creating custom targets
- Using the drug design laboratory
- Running optimization
- Testing optimized packs

