# Neuromod-LLM: Psychoactive Substance Analogues for Large Language Models

A research framework for applying neuromodulatory effects to large language models, enabling systematic investigation of how different "drug-like" interventions affect AI behavior and cognition.

## 🧠 What is Neuromod-LLM?

Neuromod-LLM is a comprehensive research platform that simulates psychoactive substance effects on AI language models. Think of it as a "digital pharmacology lab" where researchers can:

- **Apply "drug-like" effects** to language models
- **Study behavioral changes** systematically
- **Maintain experimental rigor** with blind testing
- **Generate publication-quality** statistical analysis
- **Explore AI cognition** through controlled interventions

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cneckar/neuromod-llm-poc.git
cd neuromod-llm-poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Start interactive chat with neuromodulation
python demo/chat.py

# Start the API server
cd api && python server.py
# Run statistical analysis
python -m neuromod.testing.test_runner --statistical-analysis --test sdq --treatment-packs caffeine

# List available packs
python -m neuromod.testing.test_runner --list-packs

# Verify blinding
python -m neuromod.testing.test_runner --verify-blinding
```

## 📚 Usage Modes & Documentation

This project supports several usage modes, each with detailed documentation:

### 🔬 **Research & Testing Mode**
**For**: Academic researchers, AI safety researchers, cognitive science studies
- **What**: Run psychometric tests, statistical analysis, blind experiments
- **How**: Use the testing framework and statistical analysis pipeline
- **Details**: See [`neuromod/testing/README.md`](neuromod/testing/README.md)

### 💬 **Interactive Chat Mode**
**For**: Exploratory research, demonstrations, effect testing
- **What**: Real-time neuromodulation through chat interface
- **How**: Use the interactive chat system with pack switching
- **Details**: See [`demo/README.md`](demo/README.md)

### 🧪 **Development & Extension Mode**
**For**: Developers, researchers adding new effects/packs
- **What**: Create custom effects, build new packs, extend functionality
- **How**: Use the effects system and pack management tools
- **Details**: See [`neuromod/README.md`](neuromod/README.md) and [`tests/README.md`](tests/README.md)

### 📊 **Statistical Analysis Mode**
**For**: Data analysis, publication preparation, research validation
- **What**: Comprehensive statistical testing, effect size analysis, visualization
- **How**: Use the statistical analysis pipeline and reporting tools
- **Details**: See [`neuromod/testing/README.md`](neuromod/testing/README.md)

## 🎯 Key Capabilities

### **Neuromodulation System**
- **82 Predefined Packs**: Realistic simulations of various psychoactive substances
- **100+ Individual Effects**: Temperature, attention, steering, memory, activation, MoE, objective, input
- **Custom Combinations**: Build unique neuromodulation profiles
- **Real-time Application**: Apply effects during model inference

### **Research Framework**
- **Psychometric Tests**: PDQ-S, SDQ, DDQ, DiDQ, EDQ, CDQ, PCQ-POP, ADQ
- **Blind Testing**: Maintain experimental rigor with automated verification
- **Statistical Analysis**: Publication-quality testing and visualization
- **Effect Size Analysis**: Cohen's d, power analysis, multiple comparison correction

### **Interactive Tools**
- **Chat Interface**: Real-time pack switching and effect modification
- **Pack Management**: Load, save, and modify configurations
- **Effect Registry**: Browse and combine individual effects
- **Export/Import**: Save custom combinations and configurations

## 📦 What's Included

### **Core System**
- **`neuromod/`**: Core neuromodulation engine and effects
- **`packs/`**: 82 predefined neuromodulation configurations
- **`demo/`**: Interactive demonstrations and chat interface
- **`tests/`**: Comprehensive testing framework and validation

### **Research Tools**
- **Statistical Analysis**: Mixed-effects models, ROC curves, power analysis
- **Blinding Verification**: Automated prompt analysis for experimental rigor
- **Test Suites**: Validated psychometric questionnaires for AI behavior
- **Reporting**: Publication-ready analysis and visualization

### **Documentation**
- **API Reference**: Comprehensive code documentation
- **Research Guides**: Experimental design and statistical methods
- **Tutorials**: Step-by-step usage examples
- **Examples**: Real-world research applications

## 🔬 Research Applications

### **AI Safety Research**
- Understanding how model behavior can be systematically modified
- Studying the effects of different "cognitive interventions"
- Developing frameworks for controlled AI behavior modification

### **Cognitive Science**
- Investigating AI cognition through controlled perturbations
- Studying attention, memory, and decision-making mechanisms
- Exploring the relationship between architecture and behavior

### **Drug Analogue Research**
- Simulating psychoactive substance effects on AI systems
- Studying behavioral changes in controlled environments
- Developing computational models of cognitive modification

## 🛠️ Development & Contributing

### **Adding New Effects**
```python
from neuromod.effects import BaseEffect

class CustomEffect(BaseEffect):
    def apply(self, model, inputs, **kwargs):
        # Custom effect implementation
        return modified_inputs
```

### **Creating New Packs**
```json
{
  "name": "custom_pack",
  "description": "Custom neuromodulation profile",
  "effects": [
    {
      "effect": "temperature",
      "weight": 0.5,
      "direction": "up"
    }
  ]
}
```

### **Running Tests**
```bash
# Full test suite
python tests/run_tests.py

# Specific test categories
python -m neuromod.testing.test_runner --test sdq --packs-to-apply caffeine
```

## 📚 Getting Started by Use Case

### **I want to explore neuromodulation effects**
1. Start with `python demo/chat.py`
2. See [`demo/README.md`](demo/README.md) for detailed chat commands
3. Try different packs and observe behavioral changes

### **I want to run research experiments**
1. Review [`neuromod/testing/README.md`](neuromod/testing/README.md)
2. Use `neuromod-test --help` to see available options
3. Run psychometric tests with different packs

### **I want to analyze data statistically**
1. See the statistical analysis section in [`neuromod/testing/README.md`](neuromod/testing/README.md)
2. Use `--statistical-analysis` flag with test runner
3. Generate publication-ready reports and visualizations

### **I want to develop new effects**
1. Review [`neuromod/README.md`](neuromod/README.md) for architecture details
2. Study existing effects in `neuromod/effects.py`
3. Add tests in `tests/test_effects.py`

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:

- Code contributions
- Documentation improvements
- Bug reports
- Feature requests
- Research collaborations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Research community for feedback and testing
- Open source contributors
- Academic collaborators

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/cneckar/neuromod-llm-poc/issues)
- **Discussions**: [GitHub Discussions](https://github.com/cneckar/neuromod-llm-poc/discussions)
- **Email**: [Your Email]

---

**⚠️ Research Use Only**: This framework is designed for research purposes. Please use responsibly and in accordance with applicable regulations and ethical guidelines.
