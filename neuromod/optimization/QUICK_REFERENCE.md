# Pack Optimization Quick Reference

## 🚀 Quick Commands

### CLI Usage
```bash
# Test a pack against targets
python -m neuromod.optimization.cli test-pack --target joyful_social --prompts "Hello" "How are you?"

# Run optimization
python -m neuromod.optimization.cli optimize --pack mdma --target mdma_ecstasy --method evolutionary

# Create custom target
python -m neuromod.optimization.cli create-target --name my_target --emotion joy 0.8 --behavior socialization 0.9
```

### Python API
```python
from neuromod.optimization import PackOptimizer, OptimizationConfig, OptimizationMethod
from neuromod.optimization.targets import BehavioralTarget, TargetSpec, TargetType, OptimizationObjective

# 1. Create target
target = BehavioralTarget(
    name="my_target",
    description="My custom behavioral target",
    targets=[
        TargetSpec("emotion_joy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.8, 1.0),
        TargetSpec("behavior_socialization", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.9, 1.0)
    ]
)

# 2. Configure optimization
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=20,
    population_size=15
)

# 3. Run optimization
optimizer = PackOptimizer(model_manager, evaluation_framework, config)
result = optimizer.optimize_pack(base_pack, target, test_prompts)
```

## 🎯 Target Types

| Type | Description | Examples |
|------|-------------|----------|
| `EMOTION` | Emotional states | joy, empathy, anxiety, fear, calm |
| `BEHAVIOR` | Behavioral patterns | socialization, creativity, focus, reflection |
| `LATENT_AXIS` | Dimensional states | arousal, valence, certainty, openness, sociality |
| `METRIC` | Text quality metrics | coherence, originality, thoughtfulness |

## 🔧 Optimization Methods

| Method | Best For | Speed | Exploration |
|--------|----------|-------|-------------|
| `EVOLUTIONARY` | Complex multi-objective | Medium | High |
| `BAYESIAN` | Smooth parameter spaces | Fast | Medium |
| `REINFORCEMENT_LEARNING` | Sequential decisions | Slow | High |
| `RANDOM_SEARCH` | Baseline comparison | Fast | Low |

## 📊 Preset Targets

```python
from neuromod.optimization import (
    create_joyful_social_target,
    create_creative_focused_target,
    create_calm_reflective_target
)

# Use presets
joyful_target = create_joyful_social_target()
creative_target = create_creative_focused_target()
calm_target = create_calm_reflective_target()
```

## 🧪 Example: Alcohol-like Intoxicant

```python
# Define alcohol behavioral profile
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

# Optimize
result = optimizer.optimize_pack(alcohol_pack, alcohol_target, test_prompts)
```

## 🧪 Example: MDMA-like Experience

```python
# Define MDMA behavioral profile
mdma_target = BehavioralTarget(
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
        TargetSpec("emotion_anger", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.1, 0.6)
    ]
)
```

## ⚙️ Configuration Tips

### For Quick Testing
```python
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=5,
    population_size=8,
    mutation_rate=0.4
)
```

### For Production Optimization
```python
config = OptimizationConfig(
    method=OptimizationMethod.EVOLUTIONARY,
    max_iterations=50,
    population_size=30,
    mutation_rate=0.3,
    crossover_rate=0.7,
    convergence_threshold=1e-4
)
```

### For Exploration
```python
config = OptimizationConfig(
    method=OptimizationMethod.BAYESIAN,
    max_iterations=30,
    acquisition_function="expected_improvement"
)
```

## 🔍 Debugging

### Enable Debug Output
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Emotion Tracking
```python
# The system will output detailed emotion computation logs:
# 🔍 Emotion computation debug:
#    Buffer sizes: surprisal=3, entropy=3
#    Window averages: surprisal=4.426, entropy=4.140
#    Final axes: arousal=0.382, valence=-0.316, certainty=-0.994
```

### Monitor Optimization Progress
```python
# Progress is logged automatically:
# INFO:neuromod.optimization.pack_optimizer:Iteration 5: Best loss = 0.1234
```

## 📁 File Structure

```
neuromod/optimization/
├── README.md                 # Comprehensive documentation
├── QUICK_REFERENCE.md        # This file
├── __init__.py              # Package initialization
├── targets.py               # Behavioral target definitions
├── evaluation.py            # Text and behavior evaluation
├── pack_optimizer.py        # Main optimization engine
├── probe_evaluator.py       # Real-time emotion tracking
├── laboratory.py            # Interactive sessions
├── cli.py                   # Command-line interface
├── bayesian_optimizer.py    # Bayesian optimization
├── rl_optimizer.py          # Reinforcement learning
└── evolutionary_ops.py      # Evolutionary operators
```

## 🚨 Common Issues

| Issue | Solution |
|-------|----------|
| Tensor type errors | Ensure all probe signals are properly converted to tensors |
| Missing emotion data | Check that emotion system is properly initialized |
| Convergence issues | Try different optimization methods or adjust parameters |
| Memory issues | Reduce population size or use smaller models |

## 📈 Performance Tips

1. **Use Test Mode**: Set `test_mode=True` for faster evaluation
2. **Reduce Prompts**: Use fewer test prompts for quicker runs
3. **Smaller Populations**: Start with smaller population sizes
4. **Parallel Processing**: Use multiple workers for evaluation
5. **Caching**: Reuse model instances when possible

## 🔗 Related Documentation

- [Full README](README.md) - Comprehensive optimization framework documentation
- [Core Neuromodulation](../README.md) - Main neuromodulation system
- [Testing Framework](../testing/README.md) - Research and validation tools
- [API Reference](__init__.py) - Complete API documentation
