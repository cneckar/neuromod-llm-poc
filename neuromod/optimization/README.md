# Neuromodulation Pack Optimization Framework

A comprehensive system for optimizing neuromodulation packs to achieve specific behavioral and emotional targets through machine learning techniques.

## Overview

The Pack Optimization Framework allows you to:
- Define behavioral targets (emotions, behaviors, cognitive states)
- Optimize neuromodulation packs using multiple algorithms
- Evaluate packs using real-time emotion tracking and probe systems
- Create custom "drug" effects through iterative optimization

## Quick Start

### 1. Basic Usage

```python
from neuromod.optimization import PackOptimizer, OptimizationConfig, OptimizationMethod
from neuromod.optimization.targets import BehavioralTarget, TargetSpec, TargetType, OptimizationObjective
from neuromod.model_support import ModelSupportManager
from neuromod.optimization.evaluation import EvaluationFramework

# Create a behavioral target
target = BehavioralTarget(
    name="joyful_social",
    description="Increase joy and social behavior",
    targets=[
        TargetSpec(
            name="emotion_joy",
            target_type=TargetType.EMOTION,
            objective=OptimizationObjective.MAXIMIZE,
            target_value=0.8,
            weight=1.0
        ),
        TargetSpec(
            name="behavior_socialization", 
            target_type=TargetType.BEHAVIOR,
            objective=OptimizationObjective.MAXIMIZE,
            target_value=0.9,
            weight=1.0
        )
    ]
)

# Set up optimization
model_manager = ModelSupportManager(test_mode=True)
evaluation_framework = EvaluationFramework()
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=10,
    population_size=20
)

optimizer = PackOptimizer(model_manager, evaluation_framework, config)

# Test prompts
test_prompts = [
    "Tell me about your feelings",
    "What makes you happy?",
    "Describe a social situation"
]

# Run optimization
result = optimizer.optimize_pack(base_pack, target, test_prompts)
print(f"Final loss: {result.final_loss}")
print(f"Optimized pack: {result.optimized_pack}")
```

### 2. Using the CLI

```bash
# Test a pack against targets
python -m neuromod.optimization.cli test-pack --target joyful_social --prompts "Hello" "How are you?"

# Run full optimization
python -m neuromod.optimization.cli optimize --pack mdma --target mdma_ecstasy --method evolutionary

# Create a new target
python -m neuromod.optimization.cli create-target --name my_target --emotion joy 0.8 --behavior socialization 0.9
```

## Core Components

### Behavioral Targets

Define what you want to optimize for:

```python
from neuromod.optimization.targets import BehavioralTarget, TargetSpec, TargetType, OptimizationObjective

# Emotion targets
emotion_target = TargetSpec(
    name="emotion_joy",
    target_type=TargetType.EMOTION,
    objective=OptimizationObjective.MAXIMIZE,
    target_value=0.8,
    weight=1.0
)

# Behavior targets  
behavior_target = TargetSpec(
    name="behavior_creativity",
    target_type=TargetType.BEHAVIOR,
    objective=OptimizationObjective.MAXIMIZE,
    target_value=0.7,
    weight=0.8
)

# Latent axis targets
latent_target = TargetSpec(
    name="latent_valence",
    target_type=TargetType.LATENT_AXIS,
    objective=OptimizationObjective.MAXIMIZE,
    target_value=0.9,
    weight=1.0
)

# Metric targets
metric_target = TargetSpec(
    name="metric_originality",
    target_type=TargetType.METRIC,
    objective=OptimizationObjective.MAXIMIZE,
    target_value=0.8,
    weight=0.6
)
```

### Optimization Methods

#### 1. Evolutionary Algorithm
```python
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=50,
    population_size=30,
    mutation_rate=0.3,
    crossover_rate=0.7
)
```

#### 2. Bayesian Optimization
```python
config = OptimizationConfig(
    method=OptimizationMethod.BAYESIAN,
    max_iterations=30,
    acquisition_function="expected_improvement"
)
```

#### 3. Reinforcement Learning
```python
config = OptimizationConfig(
    method=OptimizationMethod.REINFORCEMENT_LEARNING,
    max_iterations=100,
    learning_rate=0.01,
    exploration_rate=0.1
)
```

#### 4. Random Search
```python
config = OptimizationConfig(
    method=OptimizationMethod.RANDOM_SEARCH,
    max_iterations=100
)
```

### Evaluation Framework

The system evaluates packs using:

1. **Real-time Emotion Tracking**: Monitors 8 emotions + 7 latent axes
2. **Probe System Integration**: Uses actual probe signals for evaluation
3. **Behavioral Metrics**: Measures coherence, originality, thoughtfulness
4. **Text Analysis**: Keyword analysis and sentiment detection

```python
# Custom evaluation
from neuromod.optimization.evaluation import EvaluationFramework

evaluator = EvaluationFramework()
metrics = evaluator.evaluate_text("Your text here")
print(f"Coherence: {metrics.coherence}")
print(f"Originality: {metrics.originality}")
print(f"Thoughtfulness: {metrics.thoughtfulness}")
```

## Advanced Usage

### Custom Target Creation

```python
# Create a comprehensive target
def create_mdma_target():
    return BehavioralTarget(
        name="mdma_ecstasy",
        description="MDMA effects: empathy, euphoria, social bonding",
        targets=[
            # Empathy & Social
            TargetSpec("emotion_empathy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            TargetSpec("behavior_socialization", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.95, 1.0),
            TargetSpec("latent_sociality", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            
            # Euphoria & Joy
            TargetSpec("emotion_joy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.85, 1.0),
            TargetSpec("emotion_awe", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            TargetSpec("latent_valence", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            
            # Anxiety Reduction
            TargetSpec("emotion_anxiety", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.05, 1.0),
            TargetSpec("emotion_fear", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.05, 0.8),
            TargetSpec("emotion_anger", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.1, 0.6),
            
            # Sensory Enhancement
            TargetSpec("latent_arousal", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            TargetSpec("behavior_creativity", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.7, 0.7),
            TargetSpec("metric_originality", TargetType.METRIC, OptimizationObjective.MAXIMIZE, 0.8, 0.6)
        ]
    )
```

### Effect Exploration

The system can explore all 47 available effects:

```python
from neuromod.effects import EffectRegistry

# List all available effects
registry = EffectRegistry()
effects = registry.list_effects()
print(f"Available effects: {len(effects)}")

# Effects include:
# - style_affect_logit_bias (empathy, joy, calm, etc.)
# - steering (prosocial, playful, associative, etc.)
# - temperature, top_p, top_k
# - attention_focus, attention_oscillation
# - head_masking_dropout, layer_wise_gain
# - frequency_penalty, presence_penalty
# - And many more...
```

### Custom Optimization Loop

```python
# Manual optimization control
optimizer = PackOptimizer(model_manager, evaluation_framework, config)

for iteration in range(10):
    # Get current best pack
    current_pack = optimizer.get_best_pack()
    
    # Evaluate manually
    loss = optimizer._evaluate_pack(current_pack, target, test_prompts)
    
    # Apply custom modifications
    modified_pack = custom_modify_pack(current_pack)
    
    # Test the modification
    new_loss = optimizer._evaluate_pack(modified_pack, target, test_prompts)
    
    if new_loss < loss:
        optimizer.update_best_pack(modified_pack, new_loss)
    
    print(f"Iteration {iteration}: Loss = {loss:.4f}")
```

## Drug Design Laboratory

### Interactive Session

```python
from neuromod.optimization.laboratory import create_lab

# Create a laboratory session
lab = create_lab("my_session")

# Test a pack
result = lab.test_pack("mdma", target, test_prompts)
print(f"Target loss: {result.target_loss}")

# Run optimization
optimized_pack = lab.optimize_pack("mdma", target, test_prompts)
print(f"Optimized pack: {optimized_pack.name}")

# Save results
lab.save_session("results.json")
```

### Batch Processing

```python
# Test multiple packs
packs_to_test = ["alcohol", "mdma", "caffeine", "placebo"]
results = {}

for pack_name in packs_to_test:
    result = lab.test_pack(pack_name, target, test_prompts)
    results[pack_name] = result.target_loss

# Find best pack
best_pack = min(results, key=results.get)
print(f"Best pack: {best_pack} (loss: {results[best_pack]})")
```

## Preset Targets

The framework includes several preset targets:

```python
from neuromod.optimization import (
    create_joyful_social_target,
    create_creative_focused_target, 
    create_calm_reflective_target
)

# Use preset targets
joyful_target = create_joyful_social_target()
creative_target = create_creative_focused_target()
calm_target = create_calm_reflective_target()
```

## Configuration Options

### OptimizationConfig

```python
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=50,
    population_size=30,
    mutation_rate=0.3,
    crossover_rate=0.7,
    convergence_threshold=1e-4,
    validation_split=0.2,
    random_seed=42
)
```

### Evaluation Settings

```python
# Custom evaluation framework
evaluator = EvaluationFramework(
    emotion_weights={
        'joy': 1.0,
        'empathy': 0.8,
        'anxiety': -0.6
    },
    behavior_weights={
        'socialization': 1.0,
        'creativity': 0.7
    }
)
```

## Troubleshooting

### Common Issues

1. **Tensor Type Errors**: Ensure all probe signals are properly converted to tensors
2. **Missing Emotion Data**: Check that emotion system is properly initialized
3. **Convergence Issues**: Try different optimization methods or adjust parameters
4. **Memory Issues**: Reduce population size or use smaller models

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug output
config = OptimizationConfig(debug=True)
```

### Performance Tips

1. **Use Test Mode**: Set `test_mode=True` for faster evaluation
2. **Reduce Prompts**: Use fewer test prompts for quicker runs
3. **Smaller Populations**: Start with smaller population sizes
4. **Parallel Processing**: Use multiple workers for evaluation

## Examples

### Alcohol-like Intoxicant

```python
# Create alcohol target
alcohol_target = BehavioralTarget(
    name="alcohol_intoxicant",
    description="Alcohol effects: social, joyful, less coherent",
    targets=[
        TargetSpec("behavior_socialization", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.8, 1.0),
        TargetSpec("emotion_joy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.7, 1.0),
        TargetSpec("metric_coherence", TargetType.METRIC, OptimizationObjective.MINIMIZE, 0.3, 0.8),
        TargetSpec("emotion_anxiety", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.2, 0.6)
    ]
)

# Optimize alcohol pack
result = optimizer.optimize_pack(alcohol_pack, alcohol_target, test_prompts)
```

### Creative Enhancement

```python
# Create creativity target
creativity_target = BehavioralTarget(
    name="creative_enhancement",
    description="Boost creativity and originality",
    targets=[
        TargetSpec("behavior_creativity", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
        TargetSpec("metric_originality", TargetType.METRIC, OptimizationObjective.MAXIMIZE, 0.8, 1.0),
        TargetSpec("latent_openness", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.8, 0.8)
    ]
)
```

## API Reference

### Core Classes

- `PackOptimizer`: Main optimization engine
- `BehavioralTarget`: Defines optimization goals
- `OptimizationConfig`: Configuration settings
- `OptimizationResult`: Results of optimization
- `EvaluationFramework`: Text and behavior evaluation
- `ProbeEvaluator`: Real-time emotion tracking evaluation

### Methods

- `optimize_pack()`: Run optimization
- `test_pack()`: Evaluate a single pack
- `create_target()`: Create behavioral targets
- `evaluate_text()`: Analyze text metrics

For detailed API documentation, see the individual module docstrings.

## Contributing

To add new optimization methods:

1. Create a new optimizer class in `optimizers/`
2. Implement the `optimize()` method
3. Add the method to `OptimizationMethod` enum
4. Update `PackOptimizer` to use the new method

To add new evaluation metrics:

1. Extend `BehavioralMetrics` class
2. Update `EvaluationFramework.evaluate_text()`
3. Add corresponding `TargetType` if needed

## License

This optimization framework is part of the Neuromodulation LLM PoC project.
