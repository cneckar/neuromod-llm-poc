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

## Requirements Files Overview

| File | Purpose | Use Case |
|------|---------|----------|
| `requirements.txt` | **Complete production dependencies** | Full project installation |
| `requirements-minimal.txt` | **Essential dependencies only** | Basic functionality |
| `requirements-dev.txt` | **Development + production** | Development and testing |
| `api/requirements.txt` | **API-specific dependencies** | API server only |
| `vertex_container/requirements.txt` | **Vertex AI container** | Cloud deployment |

## Dependency Analysis by Component

### 🧠 Core Neuromodulation System
- **`torch>=2.2.0`** - PyTorch for deep learning models
- **`transformers>=4.42.0`** - Hugging Face transformers library
- **`accelerate>=0.20.0`** - Hugging Face accelerate for optimization
- **`bitsandbytes>=0.41.0`** - Quantization support for large models

### 📊 Scientific Computing
- **`numpy>=1.25.0`** - Numerical computing and array operations
- **`scipy>=1.11.0`** - Scientific computing and statistics
- **`scikit-learn>=1.3.0`** - Machine learning utilities
- **`pandas>=1.2.0`** - Data manipulation and analysis

### 📈 Visualization
- **`matplotlib>=3.7.0`** - Basic plotting and charts
- **`seaborn>=0.12.0`** - Statistical plotting and visualizations

### 🌐 Web Frameworks
- **`fastapi>=0.104.0`** - FastAPI for API server
- **`uvicorn[standard]>=0.24.0`** - ASGI server for FastAPI
- **`pydantic>=2.5.0`** - Data validation and settings
- **`streamlit>=1.28.0`** - Streamlit for web interface
- **`flask>=2.3.0`** - Flask for Vertex AI container
- **`gunicorn>=21.2.0`** - WSGI server for Flask

### ☁️ Google Cloud (Optional)
- **`google-cloud-aiplatform>=1.35.0`** - Vertex AI integration
- **`google-cloud-storage>=2.10.0`** - Google Cloud Storage
- **`google-auth>=2.17.0`** - Google authentication

### 🔧 Utilities
- **`tqdm>=4.65.0`** - Progress bars for long operations
- **`requests>=2.31.0`** - HTTP requests for API calls
- **`psutil>=5.9.0`** - System monitoring and resource usage
- **`python-multipart>=0.0.6`** - File upload support
- **`python-dotenv>=1.0.0`** - Environment variable loading

### 🧪 Testing (Development)
- **`pytest>=7.0.0`** - Testing framework
- **`pytest-cov>=4.0.0`** - Coverage reporting
- **`pytest-asyncio>=0.21.0`** - Async testing support
- **`httpx>=0.25.0`** - HTTP client for testing

### 🎨 Code Quality (Development)
- **`black>=23.0.0`** - Code formatting
- **`flake8>=6.0.0`** - Linting
- **`mypy>=1.0.0`** - Type checking
- **`isort>=5.12.0`** - Import sorting

## Installation Scenarios

### Scenario 1: Basic Research
```bash
# Install minimal requirements for basic neuromodulation
pip install -r requirements-minimal.txt

# Run basic demos
python demo/chat.py
```

### Scenario 2: Full Local Development
```bash
# Install all production dependencies
pip install -r requirements.txt

# Start API server
cd api && python server.py

# Start web interface
streamlit run web_interface.py
```

### Scenario 3: Development with Testing
```bash
# Install development requirements
pip install -r requirements-dev.txt

# Run tests
./test --quick
./test --full-stack
```

### Scenario 4: Vertex AI Deployment
```bash
# Install production requirements
pip install -r requirements.txt

# Deploy to Vertex AI
cd vertex_container
bash deploy_vertex_ai.sh deploy
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

## Common Issues and Solutions

### PyTorch Installation Issues
```bash
# For CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Google Cloud Issues
```bash
# Install Google Cloud dependencies separately
pip install google-cloud-aiplatform google-cloud-storage google-auth

# Set up authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

### Streamlit Issues
```bash
# Install Streamlit separately if needed
pip install streamlit

# Run with specific port
streamlit run web_interface.py --server.port 8501
```

### Memory Issues
```bash
# Install minimal version
pip install neuromod-llm[core]
```

### CUDA Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### Dependency Conflicts
```bash
# Create clean environment
python -m venv clean-env
source clean-env/bin/activate  # On Windows: clean-env\Scripts\activate
pip install neuromod-llm
```

## Dependency Compatibility

### Python Version Support
- **Python 3.8+** - All dependencies supported
- **Python 3.9+** - Recommended for best performance
- **Python 3.11+** - Latest features and optimizations

### Operating System Support
- **macOS** - All dependencies supported
- **Linux** - All dependencies supported
- **Windows** - Most dependencies supported (some limitations)

### GPU Support
- **CUDA 11.8+** - PyTorch GPU support
- **CPU Only** - All functionality available (slower)

## Recommendations

### For End Users
```bash
# Start with minimal requirements
pip install -r requirements-minimal.txt

# Add web interface if needed
pip install streamlit fastapi uvicorn
```

### For Developers
```bash
# Install development requirements
pip install -r requirements-dev.txt

# This includes all production + development tools
```

### For Production Deployment
```bash
# Install production requirements
pip install -r requirements.txt

# Ensure Google Cloud dependencies if using Vertex AI
pip install google-cloud-aiplatform google-cloud-storage google-auth
```

## Updating Dependencies

### Check for Updates
```bash
# Check outdated packages
pip list --outdated

# Update specific packages
pip install --upgrade torch transformers fastapi
```

### Pin Versions
```bash
# Generate requirements with exact versions
pip freeze > requirements-exact.txt

# Install exact versions
pip install -r requirements-exact.txt
```

## Getting Help

- **Documentation**: https://pihk.ai
- **Issues**: https://github.com/cneckar/neuromod-llm-poc/issues
- **Discussions**: https://github.com/cneckar/neuromod-llm-poc/discussions

## Next Steps

After installation, check out:
- [Quick Start Guide](reproduction_guide.md)
- [Troubleshooting Guide](troubleshooting.md)
- [Methodology Guide](methodology.md)

