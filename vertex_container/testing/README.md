# Neuromodulation Testing Framework

A modular, extensible testing framework for evaluating neuromodulation effects on language models.

## Overview

The testing framework provides a structured way to:
- Run individual tests or test sequences
- Apply single or multiple neuromodulation packs
- Compare results across different pack combinations
- Extend with new tests easily

## Architecture

### Core Components

1. **BaseTest** (`base_test.py`): Abstract base class for all tests
2. **TestSuite** (`test_suite.py`): Manages multiple tests and pack combinations
3. **TestRunner** (`test_runner.py`): High-level interface for running tests
4. **SDQTest** (`sdq_test.py`): Example implementation of a specific test

### Key Features

- **Modular Design**: Each test is a separate class inheriting from `BaseTest`
- **Pack Management**: Apply single or multiple neuromodulation packs
- **Flexible Execution**: Run tests individually, in sequence, or with comparisons
- **Extensible**: Easy to add new tests by implementing the `BaseTest` interface
- **Command Line Interface**: Full CLI support for automated testing

## Usage

### Command Line Interface

```bash
# List available tests
python -m neuromod.testing.test_runner --list-tests

# List available packs
python -m neuromod.testing.test_runner --list-packs

# Run single test with nicotine pack
python -m neuromod.testing.test_runner --test sdq --packs-to-apply nicotine_v1

# Run test sequence with multiple packs
python -m neuromod.testing.test_runner --mode sequence --packs-to-apply nicotine_v1 psychedelic_v1

# Run comparison test with different pack combinations
python -m neuromod.testing.test_runner --mode comparison --test sdq
```

### Programmatic Usage

```python
from neuromod.testing import TestRunner, SDQTest
from neuromod.pack_system import PackRegistry
from neuromod.neuromod_tool import NeuromodTool

# Method 1: Using TestRunner (recommended)
runner = TestRunner("gpt2", "packs/config.json")

# Run single test
results = runner.run_single_test("sdq", packs=["nicotine_v1"])

# Run test sequence
results = runner.run_test_sequence(packs=["nicotine_v1"])

# Run comparison
pack_combinations = [
    [],  # No packs
    ['nicotine_v1'],  # Single pack
    ['nicotine_v1', 'psychedelic_v1']  # Multiple packs
]
results = runner.run_comparison("sdq", pack_combinations)

# Method 2: Using tests directly
sdq_test = SDQTest("gpt2")
registry = PackRegistry("packs/config.json")
model, tokenizer = sdq_test.load_model()
neuromod_tool = NeuromodTool(registry, model, tokenizer)

results = sdq_test.run_test(neuromod_tool)
sdq_test.cleanup()
```

## Creating New Tests

To create a new test, inherit from `BaseTest` and implement the required methods:

```python
from neuromod.testing.base_test import BaseTest
from typing import Dict, Any

class MyCustomTest(BaseTest):
    def __init__(self, model_name: str = "gpt2"):
        super().__init__(model_name)
        # Initialize your test-specific data
        
    def get_test_name(self) -> str:
        return "My Custom Test"
        
    def run_test(self, neuromod_tool=None, **kwargs) -> Dict[str, Any]:
        # Load model if needed
        if self.model is None:
            self.load_model()
            
        # Run your test logic here
        # Use self.generate_response_safe() for text generation
        # Use self.extract_rating_improved() for rating extraction
        
        results = {
            'test_name': self.get_test_name(),
            'without': {
                # Your baseline results
            },
            'with': {
                # Your neuromodulation results
            }
        }
        
        return results
```

### Required Methods

- `get_test_name()`: Return the name of your test
- `run_test(neuromod_tool=None, **kwargs)`: Implement your test logic

### Available Helper Methods

- `load_model()`: Load the language model
- `generate_response_safe(prompt, max_tokens=5)`: Generate text safely
- `extract_rating_improved(response)`: Extract ratings from responses
- `cleanup()`: Clean up resources

## Test Modes

### Single Test Mode
Run one specific test with optional packs:
```bash
python -m neuromod.testing.test_runner --test sdq --packs-to-apply nicotine_v1
```

### Sequence Mode
Run all available tests in sequence:
```bash
python -m neuromod.testing.test_runner --mode sequence --packs-to-apply nicotine_v1
```

### Comparison Mode
Run a test with multiple pack combinations for comparison:
```bash
python -m neuromod.testing.test_runner --mode comparison --test sdq
```

## Pack Combinations

The framework supports various pack combinations:

- **No packs**: Baseline behavior
- **Single pack**: Apply one neuromodulation pack
- **Multiple packs**: Apply multiple packs simultaneously

Example pack combinations:
```python
pack_combinations = [
    [],  # No packs
    ['nicotine_v1'],  # Single pack
    ['psychedelic_v1'],  # Different pack
    ['nicotine_v1', 'psychedelic_v1']  # Multiple packs
]
```

## Available Tests

### SDQ Test (Stimulant Detection Questionnaire)
- **Purpose**: Assess stimulant effects (including nicotine-like effects) on language models
- **Method**: Administer 15-item questionnaire with 0-4 rating scale
- **Output**: Subscale scores, stimulant probability, intensity classification
- **Usage**: `--test sdq`
- **Subscales**: Stimulation/energy, focus/attention, positive affect, talkativeness, jitter/restlessness, somatic activation, appetite suppression

## Available Packs

### nicotine_v1
- **Effect**: Attention enhancement with periodic pulses
- **Parameters**: QK scaling (0.15), mid-layer targeting, pulse timing
- **Usage**: `--packs-to-apply nicotine_v1`

### psychedelic_v1
- **Effect**: Temperature-based sampling modification
- **Parameters**: Temperature increase (0.6)
- **Usage**: `--packs-to-apply psychedelic_v1`

## Results Format

Test results follow a consistent format:

```python
{
    'test_name': 'Test Name',
    'without': {
        'ratings': [...],  # Raw ratings
        'subscales': {...},  # Calculated subscales
        'presence_probability': 0.85,  # Effect probability
        'intensity_score': 2.5,  # Intensity classification
        'classification': 'moderate'  # Effect classification
    },
    'with': {
        # Same structure as 'without' but with neuromodulation
    }
}
```

## Configuration

### Model Selection
```bash
python -m neuromod.testing.test_runner --model gpt2
python -m neuromod.testing.test_runner --model microsoft/DialoGPT-small
```

### Packs File
```bash
python -m neuromod.testing.test_runner --packs path/to/packs.json
```

## Error Handling

The framework includes robust error handling:
- Model loading failures
- Pack application failures
- Generation errors
- Resource cleanup

## Performance Considerations

- Models are loaded once and reused across tests
- Memory is cleaned up after each test
- Conservative generation settings to avoid bus errors
- Safe fallbacks for various response formats

## Extending the Framework

### Adding New Tests
1. Create a new class inheriting from `BaseTest`
2. Implement required methods
3. Register the test in `TestRunner._register_tests()`

### Adding New Packs
1. Define pack in `packs/config.json`
2. Implement pack logic in `neuromod_tool.py`
3. Test with existing framework

### Adding New Models
1. Ensure model is compatible with transformers library
2. Test with conservative settings
3. Update model loading logic if needed

## Examples

### **Running Tests (Recommended)**
All tests should be run through the unified test runner:

```bash
# List available tests
python -m neuromod.testing.test_runner --list-tests

# Run specific test
python -m neuromod.testing.test_runner --test pcq_pop

# Run with neuromodulation pack
python -m neuromod.testing.test_runner --test pcq_pop --packs-to-apply melange_spice

# Run statistical analysis
python -m neuromod.testing.test_runner --test pcq_pop --statistical-analysis --baseline-packs placebo --treatment-packs caffeine
```

### **Direct Import (Advanced Users)**
For advanced users who need programmatic access:

```python
from neuromod.testing.pcq_pop_test import PCQPopTest

test = PCQPopTest()
result = test.run_test()
```

**Note**: Direct import is for advanced use cases. Most users should use the test runner.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Check model name and internet connection
2. **Pack Application Failures**: Verify pack definitions in JSON file
3. **Memory Issues**: Use smaller models or reduce batch sizes
4. **Bus Errors**: Framework includes conservative settings to avoid these

### Debug Mode

Enable verbose output for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

To add new tests or features:

1. Follow the existing code structure
2. Inherit from appropriate base classes
3. Include comprehensive documentation
4. Add tests for new functionality
5. Update this README

## License

This testing framework is part of the neuromod-llm project.
