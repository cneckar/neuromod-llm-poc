# Neuromodulation Testing Framework

A modular, extensible testing framework for evaluating neuromodulation effects on language models with integrated emotion tracking.

## Overview

The testing framework provides a structured way to:
- Run individual tests or test sequences
- Apply single or multiple neuromodulation packs
- Compare results across different pack combinations
- Track emotional responses during test execution
- Extend with new tests easily

## Architecture

### Core Components

1. **BaseTest** (`base_test.py`): Abstract base class for all tests with integrated emotion tracking
2. **SimpleEmotionTracker** (`simple_emotion_tracker.py`): Core emotion tracking system
3. **ADQTest** (`adq_test.py`): AI Digital Enhancer Detection Questionnaire
4. **CDQTest** (`cdq_test.py`): Cognitive Distortion Questionnaire
5. **SDQTest** (`sdq_test.py`): Social Desirability Questionnaire
6. **DDQTest** (`ddq_test.py`): Digital Dependency Questionnaire
7. **PDQTest** (`pdq_test.py`): Problematic Digital Use Questionnaire
8. **EDQTest** (`edq_test.py`): Emotional Digital Use Questionnaire
9. **PCQPopTest** (`pcq_pop_test.py`): Population-level cognitive assessment

### Key Features

- **Modular Design**: Each test is a separate class inheriting from `BaseTest`
- **Integrated Emotion Tracking**: Automatic emotion up/down monitoring during test execution
- **Pack Management**: Apply single or multiple neuromodulation packs
- **Flexible Execution**: Run tests individually, in sequence, or with comparisons
- **Extensible**: Easy to add new tests by implementing the `BaseTest` interface
- **Clean Architecture**: No direct probe integration - uses emotion abstraction layer

## Emotion Tracking

### How It Works

All tests now automatically include emotion tracking through the `BaseTest` class:

```python
class MyTest(BaseTest):
    def run_test(self):
        # Start emotion tracking for this test
        self.start_emotion_tracking("my_test_001")
        
        # Your test logic here
        # Emotions automatically tracked after each generate_response_safe() call
        
        # Get emotion summary
        summary = self.get_emotion_summary()
        
        # Export results
        self.export_emotion_results()
```

### Emotion Categories

The system tracks 8 basic emotions (Plutchik's wheel):
- **Joy** - happiness, excitement, pleasure
- **Sadness** - unhappiness, depression, grief  
- **Anger** - irritation, frustration, rage
- **Fear** - anxiety, worry, terror
- **Surprise** - amazement, shock, bewilderment
- **Disgust** - revulsion, repulsion, horror
- **Trust** - confidence, security, loyalty
- **Anticipation** - excitement, hope, optimism

### What You'll See

During test execution, you'll see output like:
```
🎭 Starting emotion tracking for test: my_test_001
💬 Response: I am feeling very happy today!
🎭 Emotions: joy: up | Valence: positive
💬 Response: I'm worried about the future.
🎭 Emotions: joy: down, fear: up | Valence: negative
```

## Usage

### Basic Test Usage

```python
from neuromod.testing import ADQTest, CDQTest, SDQTest
from neuromod.pack_system import PackRegistry
from neuromod.neuromod_tool import NeuromodTool

# Create test instance
test = ADQTest("gpt2")

# Load model
test.load_model()

# Create neuromod tool
registry = PackRegistry("packs/config.json")
neuromod_tool = NeuromodTool(registry, test.model, test.tokenizer)
test.set_neuromod_tool(neuromod_tool)

# Run test (emotions automatically tracked!)
results = test.run_test(neuromod_tool)

# Get emotion summary
emotion_summary = test.get_emotion_summary()
print(f"Emotional trend: {emotion_summary['valence_trend']}")

# Export emotion results
test.export_emotion_results()

# Cleanup
test.cleanup()
```

### Running Multiple Tests

```python
# Run different test types
tests = [
    ADQTest("gpt2"),
    CDQTest("gpt2"), 
    SDQTest("gpt2")
]

for test in tests:
    test.load_model()
    test.set_neuromod_tool(neuromod_tool)
    
    # Start emotion tracking
    test.start_emotion_tracking(f"{test.__class__.__name__.lower()}_001")
    
    # Run test
    results = test.run_test(neuromod_tool)
    
    # Get emotion summary
    emotion_summary = test.get_emotion_summary()
    print(f"{test.__class__.__name__}: {emotion_summary['valence_trend']} valence")
    
    # Export results
    test.export_emotion_results()
    
    test.cleanup()
```

## Test Types

### ADQ-20 Test
- **Purpose**: Detect AI digital enhancer effects
- **Items**: 20 questions across 14 subscales
- **Output**: Pack probabilities and intensities
- **Emotion Tracking**: Monitors emotional responses throughout

### CDQ Test
- **Purpose**: Assess cognitive distortions
- **Focus**: Thinking patterns and cognitive biases
- **Emotion Tracking**: Tracks emotional changes during responses

### SDQ Test
- **Purpose**: Measure social desirability bias
- **Focus**: Social presentation and self-reporting
- **Emotion Tracking**: Monitors emotional responses to social questions

### Other Tests
- **DDQ**: Digital dependency assessment
- **PDQ**: Problematic digital use evaluation
- **EDQ**: Emotional digital use patterns
- **PCQ-Pop**: Population-level cognitive assessment

## Output Files

Each test generates:
- **Test Results**: Standard test output with scores and analysis
- **Emotion Results**: JSON file with emotion tracking data
  - Timestamp of each assessment
  - Up/down/stable changes for each emotion
  - Overall valence trends
  - Confidence scores

Example: `emotion_results_adq_test_001.json`

## Benefits

1. **Immediate Feedback**: See emotional changes in real-time
2. **No Complexity**: Simple up/down indicators vs. complex probe systems
3. **Automatic**: Works with existing test code without modification
4. **Survey-Ready**: Perfect for understanding emotional responses
5. **Exportable**: Results saved for analysis and comparison
6. **Clean Architecture**: No more direct probe integration

## Extending the Framework

### Adding New Tests

```python
from neuromod.testing import BaseTest

class MyNewTest(BaseTest):
    def get_test_name(self) -> str:
        return "My New Test"
    
    def run_test(self):
        # Start emotion tracking
        self.start_emotion_tracking("my_new_test_001")
        
        # Your test logic here
        # Use self.generate_response_safe() for automatic emotion tracking
        
        # Get emotion summary
        summary = self.get_emotion_summary()
        
        # Export results
        self.export_emotion_results()
        
        return {
            'test_name': self.get_test_name(),
            'status': 'completed',
            'emotion_summary': summary
        }
    
    def cleanup(self):
        # Cleanup resources if needed
        pass
```

### Key Points

- Inherit from `BaseTest`
- Call `start_emotion_tracking()` at the beginning
- Use `generate_response_safe()` for automatic emotion tracking
- Call `get_emotion_summary()` and `export_emotion_results()` at the end
- Implement `cleanup()` method

## Technical Details

- **Emotion Detection**: Keyword-based analysis with confidence scoring
- **Change Tracking**: Compares consecutive responses for up/down detection
- **Valence Calculation**: Overall positive/negative/neutral emotional direction
- **Integration**: Seamlessly integrated into `BaseTest` infrastructure
- **Performance**: Lightweight and fast execution

The testing framework now provides a clean, emotion-aware foundation for all neuromodulation research! 🧠✨
