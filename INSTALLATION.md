# Installation Guide

## Quick Start

### Basic Installation
```bash
pip install neuromod-llm
```

### With Optimization Features
```bash
pip install neuromod-llm[optimization]
```

### With All Features
```bash
pip install neuromod-llm[all]
```

## Development Installation

### Clone and Install
```bash
git clone https://github.com/cneckar/neuromod-llm-poc.git
cd neuromod-llm-poc
pip install -e .[dev]
```

### Install from Source
```bash
pip install git+https://github.com/cneckar/neuromod-llm-poc.git
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **Transformers**: 4.35.0+
- **Memory**: 4GB RAM (8GB recommended)
- **Storage**: 2GB free space

### Recommended Requirements
- **Python**: 3.10+
- **PyTorch**: 2.2.0+
- **Transformers**: 4.45.0+
- **Memory**: 16GB RAM
- **GPU**: CUDA-compatible (optional but recommended)
- **Storage**: 10GB free space

## Installation Options

### Option 1: Minimal Installation
For basic neuromodulation effects only:
```bash
pip install neuromod-llm[core]
```

### Option 2: Research Installation
For scientific research and analysis:
```bash
pip install neuromod-llm[research]
```

### Option 3: Full Installation
For all features including optimization:
```bash
pip install neuromod-llm[all]
```

### Option 4: Development Installation
For contributing to the project:
```bash
pip install neuromod-llm[dev]
```

## Platform-Specific Instructions

### Windows
```bash
# Install with conda (recommended)
conda create -n neuromod python=3.10
conda activate neuromod
pip install neuromod-llm

# Or with pip
pip install neuromod-llm
```

### macOS
```bash
# Install with Homebrew (recommended)
brew install python@3.10
pip install neuromod-llm

# Or with conda
conda install -c conda-forge python=3.10
pip install neuromod-llm
```

### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# Create virtual environment
python3.10 -m venv neuromod-env
source neuromod-env/bin/activate

# Install package
pip install neuromod-llm
```

## GPU Support

### CUDA Installation
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install neuromod-llm
pip install neuromod-llm
```

### MPS Support (Apple Silicon)
```bash
# PyTorch with MPS support is included by default
pip install neuromod-llm
```

## Verification

### Test Installation
```python
import neuromod
print(f"Neuromod-LLM version: {neuromod.__version__}")

# Test basic functionality
from neuromod import PackRegistry
registry = PackRegistry()
packs = registry.list_packs()
print(f"Available packs: {len(packs)}")
```

### Run Example
```bash
# Test with example script
python -c "from neuromod.demo import chat; chat.main()"
```

## Troubleshooting

### Common Issues

#### 1. PyTorch Installation Issues
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 2. Memory Issues
```bash
# Install minimal version
pip install neuromod-llm[core]
```

#### 3. CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

#### 4. Dependency Conflicts
```bash
# Create clean environment
python -m venv clean-env
source clean-env/bin/activate  # On Windows: clean-env\Scripts\activate
pip install neuromod-llm
```

### Getting Help

- **Documentation**: https://neuromod-llm.readthedocs.io
- **Issues**: https://github.com/cneckar/neuromod-llm-poc/issues
- **Discussions**: https://github.com/cneckar/neuromod-llm-poc/discussions

## Next Steps

After installation, check out:
- [Quick Start Guide](examples/01_quick_start.ipynb)
- [Basic Effects Tutorial](examples/02_basic_effects.ipynb)
- [Pack Optimization Guide](examples/03_pack_optimization.ipynb)
