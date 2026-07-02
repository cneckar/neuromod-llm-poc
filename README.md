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

# Reproduce the paper (single tiered path — no GPU needed to start)
python scripts/reproduce.py --tier 0        # CPU only: validators + committed-data figs + dry-run visual pipeline

# Start interactive chat
python demo/chat.py
```

### Reproducing the paper — one path, three ways

Everything routes through **one** playbook, `scripts/reproduce.py`, which regenerates the key
figures/tables **plus the visual dose-response collateral** and writes a `REPRODUCTION_REPORT.md`
mapping each artifact to the paper claim it supports. Full details + the artifact→claim map:
**[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)**.

```bash
# 1) Locally via the script:
python scripts/reproduce.py --tier 0                 # CPU only: committed-data figs + dry-run visual pipeline
python scripts/reproduce.py --tier 1 --seeds 16      # 1 GPU: real SDXL dose-response + gpt2 text battery
python setup_hf_credentials.py                       # (tier 2) accept the Llama license + set an HF token
HUGGINGFACE_TOKEN=hf_... python scripts/reproduce.py --tier 2   # gated Llama-3.1-8B, paper-scale (exact numbers)
python scripts/reproduce.py --tier 2 --list          # inspect the plan without running anything

# 2) Locally via the notebook, or 3) in Colab (pick a tier, Run all):
#    notebooks/reproduce_paper_colab.ipynb   (auto-detects Colab vs local)
```

> The old `reproduce_results.py` still works but now just **forwards** to `scripts/reproduce.py`
> (`--test-mode` → `--tier 1`, default → `--tier 2`).

## Documentation

📖 **Full documentation available in [`docs/`](docs/)**

- **[Installation Guide](docs/installation.md)** - Installation instructions, system requirements, and dependency management
- **[Reproduction Guide](docs/reproduction_guide.md)** - Complete step-by-step instructions for reproducing experiments (the "Golden Path" for reviewers)
- **[Methodology](docs/methodology.md)** - Detailed explanation of packs, mathematical foundations (PCA, attention scaling), and biomimetic alignment theory
- **[Statistical Analysis](docs/analysis.md)** - Guide to performing statistical analysis on endpoint results
- **[Advanced Usage](docs/advanced_usage.md)** - Pack optimization, customization, Vertex AI deployment, and extension guides
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions, including HuggingFace authentication

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
