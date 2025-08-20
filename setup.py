from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuromod-llm",
    version="0.1.0",
    description="Neuromodulatory packs for Large Language Models - A framework for applying psychoactive substance analogues to LLM behavior",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Neuromod LLM Research Team",
    author_email="",
    url="https://github.com/cneckar/neuromod-llm-poc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core ML dependencies
        "torch>=2.2.0",
        "transformers>=4.42.0",
        "accelerate>=0.20.0",
        
        # Statistical analysis and scientific computing
        "numpy>=1.25.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pandas>=1.2.0",
        
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # MCP integration
        "mcp>=1.0.0",
        
        # Additional utilities
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "full": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "neuromod": [
            "packs/*.json",
            "testing/*.py",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuromod-test=neuromod.testing.test_runner:main",
        ],
    },
    keywords=[
        "neuromodulation",
        "large language models",
        "psychoactive",
        "drug analogues",
        "behavioral modification",
        "AI safety",
        "cognitive enhancement",
        "research tools",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cneckar/neuromod-llm-poc/issues",
        "Source": "https://github.com/cneckar/neuromod-llm-poc",
        "Documentation": "https://github.com/cneckar/neuromod-llm-poc#readme",
    },
)
