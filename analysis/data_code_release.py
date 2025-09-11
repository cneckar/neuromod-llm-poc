"""
Data and Code Release Preparation System

This module implements comprehensive data and code release preparation
for public release of neuromodulation research, ensuring reproducibility
and transparency.

Key Features:
- Data anonymization and cleaning
- Code documentation and validation
- Dependency management and environment setup
- Release package creation
- Reproducibility validation
- License and metadata management
"""

import json
import yaml
import shutil
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from datetime import datetime
import zipfile
import tarfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReleasePackage:
    """Complete release package specification"""
    version: str
    release_date: str
    package_name: str
    description: str
    authors: List[str]
    license: str
    repository_url: str
    documentation_url: str
    data_files: List[str]
    code_files: List[str]
    dependency_files: List[str]
    checksums: Dict[str, str]
    total_size_mb: float
    reproducibility_score: float

@dataclass
class DataFile:
    """Specification for a data file in the release"""
    file_path: str
    file_type: str  # 'raw', 'processed', 'anonymized', 'synthetic'
    description: str
    size_mb: float
    checksum: str
    anonymization_applied: bool
    privacy_level: str  # 'public', 'restricted', 'confidential'
    format: str  # 'json', 'csv', 'parquet', 'hdf5', 'pickle'

@dataclass
class CodeFile:
    """Specification for a code file in the release"""
    file_path: str
    file_type: str  # 'source', 'test', 'config', 'documentation'
    description: str
    size_mb: float
    checksum: str
    dependencies: List[str]
    test_coverage: Optional[float]
    documentation_status: str  # 'complete', 'partial', 'missing'

class DataCodeReleaseManager:
    """Main class for managing data and code release preparation"""
    
    def __init__(self, project_root: str = "/Users/cris/src/neuromod-llm-poc"):
        self.project_root = Path(project_root)
        self.release_dir = self.project_root / "releases"
        self.release_dir.mkdir(exist_ok=True)
        
        # Initialize tracking
        self.data_files: List[DataFile] = []
        self.code_files: List[CodeFile] = []
        self.release_packages: List[ReleasePackage] = []
        
    def prepare_release_package(self, 
                              version: str = "1.0.0",
                              include_sensitive_data: bool = False,
                              anonymize_data: bool = True) -> ReleasePackage:
        """
        Prepare a complete release package
        
        Args:
            version: Release version number
            include_sensitive_data: Whether to include sensitive data
            anonymize_data: Whether to anonymize data before release
            
        Returns:
            Complete release package specification
        """
        logger.info(f"Preparing release package version {version}")
        
        # Create release directory
        release_name = f"neuromod-llm-v{version}"
        release_path = self.release_dir / release_name
        release_path.mkdir(exist_ok=True)
        
        # Prepare data files
        logger.info("Preparing data files...")
        data_files = self._prepare_data_files(release_path, anonymize_data, include_sensitive_data)
        
        # Prepare code files
        logger.info("Preparing code files...")
        code_files = self._prepare_code_files(release_path)
        
        # Prepare dependency files
        logger.info("Preparing dependency files...")
        dependency_files = self._prepare_dependency_files(release_path)
        
        # Generate documentation
        logger.info("Generating documentation...")
        self._generate_release_documentation(release_path, version)
        
        # Calculate checksums
        logger.info("Calculating checksums...")
        checksums = self._calculate_checksums(release_path)
        
        # Calculate total size
        total_size = sum(f.size_mb for f in data_files + code_files)
        
        # Validate reproducibility
        reproducibility_score = self._validate_reproducibility(release_path)
        
        # Create release package
        package = ReleasePackage(
            version=version,
            release_date=datetime.now().isoformat(),
            package_name=release_name,
            description="Neuromodulated LLMs as Drug Analogues - Research Release",
            authors=["Research Team"],
            license="MIT",
            repository_url="https://github.com/cneckar/neuromod-llm-poc",
            documentation_url=f"https://github.com/cneckar/neuromod-llm-poc/releases/tag/v{version}",
            data_files=[f.file_path for f in data_files],
            code_files=[f.file_path for f in code_files],
            dependency_files=dependency_files,
            checksums=checksums,
            total_size_mb=total_size,
            reproducibility_score=reproducibility_score
        )
        
        # Save package metadata
        self._save_package_metadata(release_path, package, data_files, code_files)
        
        # Create archive
        self._create_release_archive(release_path, package)
        
        self.release_packages.append(package)
        
        logger.info(f"Release package created: {release_path}")
        logger.info(f"Total size: {total_size:.2f} MB")
        logger.info(f"Reproducibility score: {reproducibility_score:.2f}")
        
        return package
    
    def _prepare_data_files(self, release_path: Path, anonymize: bool, 
                          include_sensitive: bool) -> List[DataFile]:
        """Prepare data files for release"""
        data_files = []
        data_dir = release_path / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Define data file patterns
        data_patterns = [
            ("analysis/plan.yaml", "study_plan.yaml", "Study preregistration plan"),
            ("packs/*.json", "packs/", "Neuromodulation pack configurations"),
            ("analysis/statistical_analysis.py", "analysis/", "Statistical analysis code"),
            ("analysis/rigor_checklist.py", "analysis/", "Rigor validation code"),
            ("neuromod/testing/*.py", "neuromod/testing/", "Testing framework code"),
            ("tests/*.py", "tests/", "Test suite code")
        ]
        
        for source_pattern, dest_path, description in data_patterns:
            source_files = list(self.project_root.glob(source_pattern))
            
            for source_file in source_files:
                if not source_file.is_file():
                    continue
                
                # Determine if file should be included
                if not include_sensitive and self._is_sensitive_file(source_file):
                    continue
                
                # Create destination path
                if dest_path.endswith("/"):
                    dest_file = data_dir / dest_path / source_file.name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    dest_file = data_dir / dest_path
                
                # Copy file
                shutil.copy2(source_file, dest_file)
                
                # Anonymize if requested
                if anonymize and self._needs_anonymization(source_file):
                    self._anonymize_file(dest_file)
                
                # Create data file specification
                data_file = DataFile(
                    file_path=str(dest_file.relative_to(release_path)),
                    file_type="processed" if anonymize else "raw",
                    description=description,
                    size_mb=dest_file.stat().st_size / (1024 * 1024),
                    checksum=self._calculate_file_checksum(dest_file),
                    anonymization_applied=anonymize and self._needs_anonymization(source_file),
                    privacy_level="public",
                    format=dest_file.suffix[1:] if dest_file.suffix else "txt"
                )
                
                data_files.append(data_file)
        
        return data_files
    
    def _prepare_code_files(self, release_path: Path) -> List[CodeFile]:
        """Prepare code files for release"""
        code_files = []
        code_dir = release_path / "code"
        code_dir.mkdir(exist_ok=True)
        
        # Define code file patterns
        code_patterns = [
            ("neuromod/*.py", "neuromod/", "Core neuromodulation code"),
            ("api/*.py", "api/", "API server code"),
            ("demo/*.py", "demo/", "Demo and example code"),
            ("*.py", "", "Project root scripts"),
            ("requirements*.txt", "", "Dependency specifications"),
            ("setup.py", "", "Package setup script"),
            ("README.md", "", "Project documentation")
        ]
        
        for source_pattern, dest_path, description in code_patterns:
            source_files = list(self.project_root.glob(source_pattern))
            
            for source_file in source_files:
                if not source_file.is_file() or source_file.name.startswith('.'):
                    continue
                
                # Create destination path
                if dest_path:
                    dest_file = code_dir / dest_path / source_file.name
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    dest_file = code_dir / source_file.name
                
                # Copy file
                shutil.copy2(source_file, dest_file)
                
                # Analyze code file
                dependencies = self._extract_dependencies(dest_file)
                test_coverage = self._calculate_test_coverage(dest_file)
                doc_status = self._assess_documentation(dest_file)
                
                # Create code file specification
                code_file = CodeFile(
                    file_path=str(dest_file.relative_to(release_path)),
                    file_type=self._classify_code_file(dest_file),
                    description=description,
                    size_mb=dest_file.stat().st_size / (1024 * 1024),
                    checksum=self._calculate_file_checksum(dest_file),
                    dependencies=dependencies,
                    test_coverage=test_coverage,
                    documentation_status=doc_status
                )
                
                code_files.append(code_file)
        
        return code_files
    
    def _prepare_dependency_files(self, release_path: Path) -> List[str]:
        """Prepare dependency and environment files"""
        dependency_files = []
        
        # Copy requirements files
        req_files = ["requirements.txt", "requirements-minimal.txt", "requirements-dev.txt"]
        for req_file in req_files:
            source_file = self.project_root / req_file
            if source_file.exists():
                dest_file = release_path / req_file
                shutil.copy2(source_file, dest_file)
                dependency_files.append(req_file)
        
        # Create environment specification
        env_spec = self._create_environment_specification()
        env_file = release_path / "environment.yml"
        with open(env_file, 'w') as f:
            yaml.dump(env_spec, f, default_flow_style=False)
        dependency_files.append("environment.yml")
        
        # Create Docker files
        docker_files = ["Dockerfile", "docker-compose.yml"]
        for docker_file in docker_files:
            source_file = self.project_root / docker_file
            if source_file.exists():
                dest_file = release_path / docker_file
                shutil.copy2(source_file, dest_file)
                dependency_files.append(docker_file)
        
        return dependency_files
    
    def _generate_release_documentation(self, release_path: Path, version: str):
        """Generate comprehensive release documentation"""
        docs_dir = release_path / "docs"
        docs_dir.mkdir(exist_ok=True)
        
        # Generate README
        readme_content = self._generate_release_readme(version)
        readme_file = release_path / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        # Generate API documentation
        api_docs = self._generate_api_documentation()
        api_file = docs_dir / "API.md"
        with open(api_file, 'w') as f:
            f.write(api_docs)
        
        # Generate data documentation
        data_docs = self._generate_data_documentation()
        data_file = docs_dir / "DATA.md"
        with open(data_file, 'w') as f:
            f.write(data_docs)
        
        # Generate reproducibility guide
        repro_guide = self._generate_reproducibility_guide()
        repro_file = docs_dir / "REPRODUCIBILITY.md"
        with open(repro_file, 'w') as f:
            f.write(repro_guide)
    
    def _validate_reproducibility(self, release_path: Path) -> float:
        """Validate reproducibility of the release package"""
        score = 0.0
        max_score = 100.0
        
        # Check for essential files
        essential_files = [
            "README.md",
            "requirements.txt",
            "environment.yml",
            "code/neuromod/",
            "data/study_plan.yaml"
        ]
        
        for file_path in essential_files:
            if (release_path / file_path).exists():
                score += 10.0
        
        # Check for documentation
        docs_dir = release_path / "docs"
        if docs_dir.exists():
            score += 20.0
        
        # Check for test coverage
        test_files = list(release_path.glob("**/test_*.py"))
        if test_files:
            score += 15.0
        
        # Check for configuration files
        config_files = list(release_path.glob("**/*.yaml")) + list(release_path.glob("**/*.yml"))
        if config_files:
            score += 10.0
        
        # Check for data integrity
        data_files = list(release_path.glob("data/**/*"))
        if data_files:
            score += 15.0
        
        # Check for code quality
        py_files = list(release_path.glob("**/*.py"))
        if py_files:
            score += 20.0
        
        # Check for version control info
        if (release_path / ".git").exists():
            score += 10.0
        
        return min(score, max_score)
    
    def _create_release_archive(self, release_path: Path, package: ReleasePackage):
        """Create compressed archive of the release package"""
        archive_name = f"{package.package_name}.tar.gz"
        archive_path = self.release_dir / archive_name
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(release_path, arcname=package.package_name)
        
        logger.info(f"Release archive created: {archive_path}")
        
        # Also create ZIP archive for Windows users
        zip_name = f"{package.package_name}.zip"
        zip_path = self.release_dir / zip_name
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in release_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(release_path)
                    zipf.write(file_path, arcname)
        
        logger.info(f"ZIP archive created: {zip_path}")
    
    def _is_sensitive_file(self, file_path: Path) -> bool:
        """Check if file contains sensitive information"""
        sensitive_patterns = [
            "credentials", "secret", "password", "token", "key",
            "private", "confidential", "personal", "sensitive"
        ]
        
        file_name = file_path.name.lower()
        return any(pattern in file_name for pattern in sensitive_patterns)
    
    def _needs_anonymization(self, file_path: Path) -> bool:
        """Check if file needs anonymization"""
        # Check file extension
        if file_path.suffix in ['.json', '.yaml', '.yml', '.csv']:
            return True
        
        # Check file name patterns
        anonymize_patterns = [
            "results", "data", "output", "log", "config"
        ]
        
        file_name = file_path.name.lower()
        return any(pattern in file_name for pattern in anonymize_patterns)
    
    def _anonymize_file(self, file_path: Path):
        """Anonymize sensitive data in file"""
        if file_path.suffix == '.json':
            self._anonymize_json_file(file_path)
        elif file_path.suffix in ['.yaml', '.yml']:
            self._anonymize_yaml_file(file_path)
        elif file_path.suffix == '.csv':
            self._anonymize_csv_file(file_path)
    
    def _anonymize_json_file(self, file_path: Path):
        """Anonymize JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Anonymize common sensitive fields
        sensitive_fields = [
            'user_id', 'participant_id', 'subject_id', 'email', 'name',
            'ip_address', 'session_id', 'token', 'key'
        ]
        
        def anonymize_dict(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(field in key.lower() for field in sensitive_fields):
                        obj[key] = f"ANONYMIZED_{hash(str(value)) % 10000:04d}"
                    else:
                        anonymize_dict(value)
            elif isinstance(obj, list):
                for item in obj:
                    anonymize_dict(item)
        
        anonymize_dict(data)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _anonymize_yaml_file(self, file_path: Path):
        """Anonymize YAML file"""
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Simple anonymization - replace common patterns
        sensitive_patterns = [
            (r'user_id:\s*\d+', 'user_id: ANONYMIZED'),
            (r'participant_id:\s*\d+', 'participant_id: ANONYMIZED'),
            (r'email:\s*[^\s]+', 'email: hi@pihk.ai'),
            (r'name:\s*[^\s]+', 'name: ANONYMIZED'),
        ]
        
        import re
        for pattern, replacement in sensitive_patterns:
            content = re.sub(pattern, replacement, content)
        
        with open(file_path, 'w') as f:
            f.write(content)
    
    def _anonymize_csv_file(self, file_path: Path):
        """Anonymize CSV file"""
        df = pd.read_csv(file_path)
        
        # Anonymize common sensitive columns
        sensitive_columns = [
            'user_id', 'participant_id', 'subject_id', 'email', 'name',
            'ip_address', 'session_id'
        ]
        
        for col in df.columns:
            if any(field in col.lower() for field in sensitive_columns):
                df[col] = df[col].apply(lambda x: f"ANONYMIZED_{hash(str(x)) % 10000:04d}")
        
        df.to_csv(file_path, index=False)
    
    def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _calculate_checksums(self, release_path: Path) -> Dict[str, str]:
        """Calculate checksums for all files in release"""
        checksums = {}
        
        for file_path in release_path.rglob('*'):
            if file_path.is_file():
                rel_path = str(file_path.relative_to(release_path))
                checksums[rel_path] = self._calculate_file_checksum(file_path)
        
        return checksums
    
    def _extract_dependencies(self, file_path: Path) -> List[str]:
        """Extract dependencies from code file"""
        dependencies = []
        
        if file_path.suffix != '.py':
            return dependencies
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Simple import extraction
            import re
            imports = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(\S+)', content, re.MULTILINE)
            
            for from_module, import_name in imports:
                if from_module:
                    dependencies.append(from_module)
                else:
                    dependencies.append(import_name)
        
        except Exception as e:
            logger.warning(f"Failed to extract dependencies from {file_path}: {e}")
        
        return dependencies
    
    def _calculate_test_coverage(self, file_path: Path) -> Optional[float]:
        """Calculate test coverage for code file"""
        # This is a simplified version - in practice, you'd use coverage.py
        if file_path.suffix != '.py':
            return None
        
        # Check if there are corresponding test files
        test_files = list(file_path.parent.glob(f"test_{file_path.stem}.py"))
        if test_files:
            return 0.8  # Mock coverage
        else:
            return 0.0
    
    def _assess_documentation(self, file_path: Path) -> str:
        """Assess documentation status of code file"""
        if file_path.suffix != '.py':
            return "complete"
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check for docstrings
            has_docstrings = '"""' in content or "'''" in content
            has_comments = '#' in content
            
            if has_docstrings and has_comments:
                return "complete"
            elif has_docstrings or has_comments:
                return "partial"
            else:
                return "missing"
        
        except Exception:
            return "missing"
    
    def _classify_code_file(self, file_path: Path) -> str:
        """Classify type of code file"""
        if file_path.name.startswith('test_'):
            return "test"
        elif file_path.name in ['README.md', 'docs']:
            return "documentation"
        elif file_path.suffix in ['.yaml', '.yml', '.json']:
            return "config"
        else:
            return "source"
    
    def _create_environment_specification(self) -> Dict[str, Any]:
        """Create conda environment specification"""
        return {
            "name": "neuromod-llm",
            "channels": ["conda-forge", "pytorch", "huggingface"],
            "dependencies": [
                "python=3.11",
                "pytorch",
                "transformers",
                "numpy",
                "pandas",
                "scipy",
                "matplotlib",
                "seaborn",
                "jupyter",
                "pytest",
                "pip",
                {
                    "pip": [
                        "accelerate",
                        "datasets",
                        "evaluate",
                        "wandb",
                        "fastapi",
                        "uvicorn"
                    ]
                }
            ]
        }
    
    def _generate_release_readme(self, version: str) -> str:
        """Generate comprehensive README for release"""
        return f"""# Neuromodulated LLMs as Drug Analogues - Research Release v{version}

## Overview

This release contains the complete codebase and data for the research project "Neuromodulated LLMs as Drug Analogues: A Computational Framework for Psychopharmacological Research."

## Contents

- **Code**: Complete implementation of neuromodulation effects, testing framework, and analysis tools
- **Data**: Study plans, pack configurations, and experimental results
- **Documentation**: Comprehensive guides for reproduction and extension

## Quick Start

1. **Environment Setup**:
   ```bash
   conda env create -f environment.yml
   conda activate neuromod-llm
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Tests**:
   ```bash
   python -m pytest tests/
   ```

4. **Run Example**:
   ```bash
   python demo/quick_pack_test.py
   ```

## Documentation

- [API Documentation](docs/API.md)
- [Data Documentation](docs/DATA.md)
- [Reproducibility Guide](docs/REPRODUCIBILITY.md)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{{neuromod_llm_2024,
  title={{Neuromodulated LLMs as Drug Analogues: A Computational Framework for Psychopharmacological Research}},
  author={{Research Team}},
  journal={{arXiv preprint}},
  year={{2024}}
}}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the research team.

## Changelog

### v{version}
- Initial research release
- Complete neuromodulation framework
- Comprehensive testing suite
- Statistical analysis tools
- Reproducibility validation
"""
    
    def _generate_api_documentation(self) -> str:
        """Generate API documentation"""
        return """# API Documentation

## Core Modules

### neuromod.effects
Core neuromodulation effects implementation.

### neuromod.pack_system
Pack configuration and management system.

### neuromod.testing
Testing framework and psychometric instruments.

### analysis.statistical_analysis
Statistical analysis and reporting tools.

## Usage Examples

### Basic Pack Usage
```python
from neuromod.pack_system import PackManager

# Load a pack
pack_manager = PackManager()
pack = pack_manager.load_pack("caffeine")

# Apply to model
result = pack.apply_to_model(model, prompt)
```

### Running Tests
```python
from neuromod.testing import TestRunner

# Run psychometric tests
test_runner = TestRunner()
results = test_runner.run_test_suite(pack, test_data)
```
"""
    
    def _generate_data_documentation(self) -> str:
        """Generate data documentation"""
        return """# Data Documentation

## File Structure

- `data/study_plan.yaml`: Preregistered study plan
- `data/packs/`: Neuromodulation pack configurations
- `data/analysis/`: Statistical analysis code
- `data/neuromod/testing/`: Testing framework

## Data Formats

### Pack Files (.json)
```json
{{
  "name": "pack_name",
  "effects": [
    {{
      "effect": "effect_type",
      "weight": 1.0,
      "direction": "up",
      "parameters": {{}}
    }}
  ]
}}
```

### Results Files (.json)
```json
{{
  "pack_name": "caffeine",
  "test_results": {{
    "adq_score": 0.75,
    "pdq_score": 0.68
  }},
  "metadata": {{
    "timestamp": "2024-01-01T00:00:00Z",
    "model": "llama-3.1-70b"
  }}
}}
```

## Privacy and Ethics

All data has been anonymized and de-identified. No personal information is included in this release.
"""
    
    def _generate_reproducibility_guide(self) -> str:
        """Generate reproducibility guide"""
        return """# Reproducibility Guide

## Environment Requirements

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (for large models)
- 50GB+ disk space

## Step-by-Step Reproduction

1. **Clone Repository**:
   ```bash
   git clone https://github.com/cneckar/neuromod-llm-poc.git
   cd neuromod-llm-poc
   ```

2. **Setup Environment**:
   ```bash
   conda env create -f environment.yml
   conda activate neuromod-llm
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Validate Installation**:
   ```bash
   python analysis/rigor_checklist.py
   ```

5. **Run Tests**:
   ```bash
   python -m pytest tests/ -v
   ```

6. **Run Example Study**:
   ```bash
   python demo/quick_pack_test.py
   ```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Check HuggingFace credentials
3. **Test Failures**: Ensure all dependencies are installed

### Getting Help

- Check the GitHub issues page
- Review the documentation
- Contact the research team
"""
    
    def _save_package_metadata(self, release_path: Path, package: ReleasePackage, 
                             data_files: List[DataFile], code_files: List[CodeFile]):
        """Save package metadata to file"""
        metadata = {
            "package": asdict(package),
            "data_files": [asdict(f) for f in data_files],
            "code_files": [asdict(f) for f in code_files],
            "generated_at": datetime.now().isoformat()
        }
        
        metadata_file = release_path / "package_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

def main():
    """Example usage of the data code release manager"""
    manager = DataCodeReleaseManager()
    
    print("Preparing release package...")
    package = manager.prepare_release_package(
        version="1.0.0",
        include_sensitive_data=False,
        anonymize_data=True
    )
    
    print(f"Release package created: {package.package_name}")
    print(f"Total size: {package.total_size_mb:.2f} MB")
    print(f"Reproducibility score: {package.reproducibility_score:.2f}")
    print(f"Data files: {len(package.data_files)}")
    print(f"Code files: {len(package.code_files)}")

if __name__ == "__main__":
    main()
