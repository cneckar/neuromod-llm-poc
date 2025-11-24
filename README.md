# Neuromodulated Language Models

**Biomimetic Alignment: Borrowing Control Primitives from Biology to Stabilize AI**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We introduce and evaluate a suite of inference-time "neuromodulation packs" that borrow control primitives from biological gain control mechanisms to stabilize and steer large language models. Using a double-blind, placebo-controlled, within-model crossover design, we benchmark behavioral signatures against human subjective-effect profiles. Our results demonstrate **biomimetic alignment** between biological control theory and computational inference steering.

**Key Findings:**
- Serotonergic Agonist class (LSD, Psilocybin) reliably induced high-entropy altered states detectable via the novel *Psychedelic Detection Questionnaire* (PDQ-S)
- Stimulant packs revealed a "focus ceiling" effect in reinforcement-learned models
- Biomimetic control primitives enable dynamic, reversible AI alignment without retraining

## Quick Start

```bash
# Clone repository
git clone https://github.com/cneckar/neuromod-llm-poc.git
cd neuromod-llm-poc

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Set up Hugging Face credentials (required for Llama models)
python setup_hf_credentials.py

# Reproduce paper results (default: Llama-3.1-8B, n=126)
python reproduce_results.py

# Or quick validation (test mode: fast model, small sample)
python reproduce_results.py --test-mode

# Start interactive chat
python demo/chat.py
```

## Documentation

📖 **Full documentation available in [`docs/`](docs/)**

- **[Reproduction Guide](docs/reproduction_guide.md)** - Complete step-by-step instructions for reproducing experiments (the "Golden Path" for reviewers)
- **[Methodology](docs/methodology.md)** - Detailed explanation of packs, mathematical foundations (PCA, attention scaling), and biomimetic alignment theory
- **[Advanced Usage](docs/advanced_usage.md)** - Pack optimization, customization, and extension guides
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

## What is This?

Neuromod-LLM is a research framework that applies **biomimetic control primitives** to large language models. We borrow control theory from biological neuromodulation (e.g., how serotonin or norepinephrine modulate neural circuits) and translate it into computational interventions for transformer architectures.

**We are NOT:**
- Simulating a brain
- Simulating neurotransmitters
- Creating identical mechanisms to biology

**We ARE:**
- Borrowing control theory from biological gain control mechanisms
- Implementing linear algebra operations (vector addition, attention scaling, cache decay)
- Using neuromorphic control primitives to stabilize AI
- Demonstrating biomimetic alignment (functional similarity in control dynamics)

## Key Features

- **28+ Neuromodulation Packs**: Pre-configured interventions for stimulants, psychedelics, depressants, and more
- **100+ Individual Effects**: Temperature, attention, steering, memory, activation modifications
- **Blind Experimental Design**: Double-blind, placebo-controlled, within-model crossover protocol
- **Psychometric Testing**: PDQ-S, ADQ-20, PCQ-POP-20, and other validated instruments
- **Statistical Analysis**: Mixed-effects models, FDR correction, effect size calculations
- **Pack Optimization**: Machine learning framework for optimizing behavioral targets

## Requirements

- **Python 3.8+**
- **Local Model Access**: API models (OpenAI, Anthropic) are NOT supported - requires access to model internals
- **Supported Models**: openai/gpt-oss-120b, openai/gpt-oss-20b, Llama-3.1-70B/8B, Qwen-2.5-Omni-7B, Mixtral-8×22B, and other open-source models via HuggingFace
- **GPU Recommended**: For larger models (70B+ parameters)

## Citation

If you use this work in your research, please cite:

```bibtex
@article{neuromodulated2024,
  title={Neuromodulated Language Models: Prototyping Pharmacological Analogues and Blind, Placebo-Controlled Evaluation},
  author={AiHKAL},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) or open an issue for discussion.

## Research Use Only

⚠️ **This framework is designed for research purposes.** Please use responsibly and in accordance with applicable regulations and ethical guidelines.

---

**For detailed documentation, see [`docs/`](docs/)**
