# Testing Framework

This directory contains comprehensive tests for the neuromodulation framework.

## Test Structure

### Core Tests
- **PDQ Test**: Psychedelic Detection Questionnaire
- **SDQ Test**: Stimulant Detection Questionnaire (covers nicotine-like effects)
- **DDQ Test**: Depressant/Sedative Detection Questionnaire
- **DiDQ Test**: Dissociative Detection Questionnaire
- **EDQ Test**: Empathogen/Entactogen Detection Questionnaire
- **CDQ Test**: Cannabinoid Detection Questionnaire
- **PCQ-POP Test**: Pop-Culture Pack Detection Questionnaire
- **ADQ Test**: AI Digital Enhancer Detection Questionnaire

### Integration Tests
- **Model Loading**: Tests for various model types
- **Pack Application**: Tests for neuromodulation pack application
- **Effect System**: Tests for individual effects
- **Tool Integration**: Tests for MCP tool functionality

## Test Coverage

### Neuromodulation Effects
- **Temperature Effects**: Up/down temperature modulation
- **Top-P Effects**: Nucleus sampling parameter adjustment
- **Steering Effects**: Activation vector steering
- **Attention Effects**: Head masking and QK scaling
- **KV-Cache Effects**: Memory manipulation and decay
- **Pulse Effects**: Periodic effect application

### Pack System
- **Pack Loading**: JSON configuration parsing
- **Pack Application**: Effect application and cleanup
- **Intensity Scaling**: Weight adjustment based on intensity
- **Scheduling**: Time-based effect application

### Testing Framework
- **SDQ and PDQ test integration**: Comprehensive stimulant and psychedelic effect detection
- **Statistical Analysis**: Subscale calculation and probability estimation
- **Result Aggregation**: Multi-set result combination
- **Error Handling**: Robust error recovery and reporting

## �� Running Tests

### Quick Test Suite
```bash
python tests/run_tests.py --quick
```
Runs critical functionality tests only (~30 seconds)

### Full Test Suite
```bash
python tests/run_tests.py
```
Runs all tests with detailed reporting (~2-3 minutes)

### Verbose Output
```bash
python tests/run_tests.py --verbose
```
Shows detailed test output and tracebacks

### Specific Test File
```bash
python tests/run_tests.py --test tests/test_core.py
```
Runs only the core functionality tests

### Coverage Report
```bash
python tests/run_tests.py --coverage
```
Shows what components are being tested

## 📊 Test Categories

### 1. Core System Tests (`test_core.py`)
Tests the fundamental building blocks:
- **EffectConfig**: Parameter validation, weight clamping, serialization
- **Pack**: Pack creation, effect validation, JSON conversion
- **PackRegistry**: Config loading, pack management, persistence
- **PackManager**: Effect application, cleanup, lifecycle management
- **EffectRegistry**: Effect creation, parameter handling

### 2. Effects Tests (`test_effects.py`)
Tests all 38 neuromodulation effects:
- **Initialization**: Proper parameter handling and validation
- **Application**: Effect application to models
- **Cleanup**: Proper resource cleanup
- **Logits Processing**: Sampler effects produce valid processors
- **Integration**: Multiple effects working together

### 3. Integration Tests (`test_integration.py`)
Tests complete system integration:
- **NeuromodTool**: Complete tool functionality
- **Model Integration**: Real model loading and generation
- **Pack Application**: Multi-pack scenarios
- **End-to-End**: Complete workflows
- **Testing Framework**: NDQ and PDQ test integration

## 🧪 Test Features

### ✅ Comprehensive Validation
- **Parameter Validation**: All effect parameters validated
- **Weight Clamping**: Weights properly clamped to [0, 1]
- **Direction Handling**: Up/down directions properly applied
- **Error Handling**: Graceful handling of invalid inputs

### ✅ Mock Testing
- **Model Mocking**: Tests don't require real model loading
- **Tokenizer Mocking**: Simulated tokenizer behavior
- **Effect Mocking**: Isolated effect testing

### ✅ Real Model Testing
- **GPT-2 Integration**: Tests with actual GPT-2 model
- **Generation Testing**: Real text generation with effects
- **Memory Management**: Proper cleanup and resource management

### ✅ Edge Cases
- **Invalid Packs**: Handling of malformed pack definitions
- **Missing Effects**: Graceful handling of unknown effects
- **Empty Configs**: Handling of empty configuration files
- **Concurrent Application**: Multiple pack applications

## 📈 Test Metrics

### Performance
- **Quick Tests**: ~30 seconds
- **Full Suite**: ~2-3 minutes
- **Individual Tests**: <1 second each

### Coverage
- **Lines of Code**: 100% of core functionality
- **Functions**: 100% of public APIs
- **Edge Cases**: Comprehensive edge case testing
- **Integration**: Full system integration testing

### Reliability
- **Deterministic**: Tests produce consistent results
- **Isolated**: Tests don't interfere with each other
- **Cleanup**: Proper resource cleanup after each test
- **Mocking**: No external dependencies for unit tests

## 🔧 Test Development

### Adding New Tests
1. **Identify Component**: Determine which test file to add to
2. **Create Test Class**: Inherit from `unittest.TestCase`
3. **Write Test Methods**: Use descriptive method names
4. **Add Assertions**: Test expected behavior
5. **Handle Edge Cases**: Test error conditions
6. **Update Coverage**: Ensure new functionality is tested

### Test Naming Convention
```python
def test_component_functionality(self):
    """Test specific functionality of component"""
    # Test implementation
```

### Test Structure
```python
class TestComponent(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        pass
    
    def tearDown(self):
        """Clean up test fixtures"""
        pass
```

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `neuromod` package is installed
2. **Mock Issues**: Check mock setup in test fixtures
3. **Model Loading**: Some tests require internet connection for model download
4. **Memory Issues**: Large models may require significant RAM

### Debug Mode
```bash
python tests/run_tests.py --verbose --test tests/test_core.py
```

### Individual Test Execution
```bash
python -m unittest tests.test_core.TestEffectConfig.test_effect_config_creation
```

## 📋 Test Results Interpretation

### Success Indicators
- ✅ All tests pass
- ✅ No memory leaks
- ✅ Proper cleanup
- ✅ Consistent results

### Failure Analysis
- ❌ **Failures**: Logic errors in implementation
- 🚨 **Errors**: Exceptions during execution
- ⚠️ **Skipped**: Tests that couldn't run (e.g., missing dependencies)

### Performance Metrics
- **Test Count**: Total number of tests run
- **Success Rate**: Percentage of passing tests
- **Execution Time**: Time to complete test suite
- **Memory Usage**: Peak memory consumption

## 🎉 Continuous Integration

The test suite is designed for CI/CD integration:
- **Fast Execution**: Quick tests for rapid feedback
- **Comprehensive Coverage**: Full suite for thorough validation
- **Clear Reporting**: Detailed output for debugging
- **Exit Codes**: Proper exit codes for CI systems

Run in CI:
```bash
python tests/run_tests.py --quick  # Fast feedback
python tests/run_tests.py          # Full validation
```
