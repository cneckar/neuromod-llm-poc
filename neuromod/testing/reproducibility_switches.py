#!/usr/bin/env python3
"""
Reproducibility Switches System

This module implements reproducibility controls for:
- Environment management (Python version, CUDA, etc.)
- Dependency pinning and validation
- Deterministic execution controls
- Cross-platform compatibility
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnvironmentInfo:
    """Information about the current environment"""
    python_version: str
    platform: str
    cuda_available: bool
    cuda_version: Optional[str]
    gpu_count: int
    memory_gb: float
    cpu_count: int
    pip_freeze: List[str]
    git_sha: Optional[str]
    git_branch: Optional[str]

@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility controls"""
    require_exact_python: bool = True
    require_cuda: bool = False
    min_memory_gb: float = 8.0
    require_git_clean: bool = True
    pin_dependencies: bool = True
    deterministic_seeds: bool = True
    cross_platform: bool = True

class ReproducibilityValidator:
    """Validates and enforces reproducibility requirements"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.config_file = self.project_root / "reproducibility_config.json"
        self.lock_file = self.project_root / "reproducibility.lock"
        
    def get_environment_info(self) -> EnvironmentInfo:
        """Gather comprehensive environment information"""
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Platform info
        platform_info = f"{platform.system()} {platform.release()} {platform.machine()}"
        
        # CUDA info
        cuda_available = False
        cuda_version = None
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
        except ImportError:
            pass
        
        # GPU count
        gpu_count = 0
        if cuda_available:
            try:
                import torch
                gpu_count = torch.cuda.device_count()
            except:
                pass
        
        # Memory info
        memory_gb = 0.0
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            # Fallback for systems without psutil
            memory_gb = 8.0  # Assume minimum
        
        # CPU count
        cpu_count = os.cpu_count() or 1
        
        # Pip freeze
        pip_freeze = []
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                                  capture_output=True, text=True, check=True)
            pip_freeze = result.stdout.strip().split('\n')
        except subprocess.CalledProcessError:
            logger.warning("Could not get pip freeze output")
        
        # Git info
        git_sha = None
        git_branch = None
        try:
            result = subprocess.run(["git", "rev-parse", "HEAD"], 
                                  capture_output=True, text=True, check=True, cwd=self.project_root)
            git_sha = result.stdout.strip()
            
            result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], 
                                  capture_output=True, text=True, check=True, cwd=self.project_root)
            git_branch = result.stdout.strip()
        except subprocess.CalledProcessError:
            logger.warning("Could not get git information")
        
        return EnvironmentInfo(
            python_version=python_version,
            platform=platform_info,
            cuda_available=cuda_available,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            memory_gb=memory_gb,
            cpu_count=cpu_count,
            pip_freeze=pip_freeze,
            git_sha=git_sha,
            git_branch=git_branch
        )
    
    def validate_environment(self, config: ReproducibilityConfig) -> Dict[str, Any]:
        """Validate current environment against requirements"""
        env_info = self.get_environment_info()
        validation_results = {
            "environment_info": asdict(env_info),
            "validation_passed": True,
            "warnings": [],
            "errors": []
        }
        
        # Check Python version
        if config.require_exact_python:
            expected_python = "3.11"  # Expected Python version
            if not env_info.python_version.startswith(expected_python):
                validation_results["errors"].append(
                    f"Python version {env_info.python_version} does not match expected {expected_python}"
                )
                validation_results["validation_passed"] = False
        
        # Check CUDA
        if config.require_cuda and not env_info.cuda_available:
            validation_results["errors"].append("CUDA is required but not available")
            validation_results["validation_passed"] = False
        
        # Check memory
        if env_info.memory_gb < config.min_memory_gb:
            validation_results["warnings"].append(
                f"Memory {env_info.memory_gb:.1f}GB is less than recommended {config.min_memory_gb}GB"
            )
        
        # Check git status
        if config.require_git_clean:
            try:
                result = subprocess.run(["git", "status", "--porcelain"], 
                                      capture_output=True, text=True, check=True, cwd=self.project_root)
                if result.stdout.strip():
                    validation_results["warnings"].append("Git working directory is not clean")
            except subprocess.CalledProcessError:
                validation_results["warnings"].append("Could not check git status")
        
        return validation_results
    
    def create_reproducibility_lock(self, config: ReproducibilityConfig) -> Dict[str, Any]:
        """Create a reproducibility lock file with current environment"""
        env_info = self.get_environment_info()
        
        lock_data = {
            "created_at": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "environment_info": asdict(env_info),
            "config": asdict(config),
            "reproducibility_hash": self._calculate_reproducibility_hash(env_info)
        }
        
        # Save lock file
        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        logger.info(f"Reproducibility lock created: {self.lock_file}")
        return lock_data
    
    def _calculate_reproducibility_hash(self, env_info: EnvironmentInfo) -> str:
        """Calculate a hash representing the reproducibility state"""
        # Create a hash based on key environment factors
        hash_input = f"{env_info.python_version}_{env_info.platform}_{env_info.cuda_version}_{len(env_info.pip_freeze)}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def validate_reproducibility(self, config: ReproducibilityConfig) -> Dict[str, Any]:
        """Validate that current environment matches reproducibility lock"""
        if not self.lock_file.exists():
            return {
                "validation_passed": False,
                "error": "No reproducibility lock file found. Run create_reproducibility_lock first."
            }
        
        # Load lock file
        with open(self.lock_file, 'r') as f:
            lock_data = json.load(f)
        
        # Get current environment
        current_env = self.get_environment_info()
        current_hash = self._calculate_reproducibility_hash(current_env)
        locked_hash = lock_data.get("reproducibility_hash", "")
        
        validation_results = {
            "validation_passed": current_hash == locked_hash,
            "current_hash": current_hash,
            "locked_hash": locked_hash,
            "environment_changes": [],
            "warnings": []
        }
        
        # Compare key environment factors
        locked_env = lock_data.get("environment_info", {})
        
        if current_env.python_version != locked_env.get("python_version"):
            validation_results["environment_changes"].append("Python version changed")
        
        if current_env.platform != locked_env.get("platform"):
            validation_results["environment_changes"].append("Platform changed")
        
        if current_env.cuda_version != locked_env.get("cuda_version"):
            validation_results["environment_changes"].append("CUDA version changed")
        
        if len(current_env.pip_freeze) != len(locked_env.get("pip_freeze", [])):
            validation_results["environment_changes"].append("Dependencies changed")
        
        if validation_results["environment_changes"]:
            validation_results["validation_passed"] = False
            validation_results["warnings"].append("Environment has changed since lock was created")
        
        return validation_results
    
    def create_requirements_lock(self) -> str:
        """Create a locked requirements file"""
        env_info = self.get_environment_info()
        
        requirements_content = f"""# Reproducibility Lock File
# Generated: {datetime.now().isoformat()}
# Python: {env_info.python_version}
# Platform: {env_info.platform}
# CUDA: {env_info.cuda_version or 'Not available'}

"""
        
        for package in env_info.pip_freeze:
            if package.strip():
                requirements_content += f"{package}\n"
        
        requirements_file = self.project_root / "requirements-lock.txt"
        with open(requirements_file, 'w') as f:
            requirements_content += f"\n# Reproducibility hash: {self._calculate_reproducibility_hash(env_info)}\n"
            f.write(requirements_content)
        
        logger.info(f"Requirements lock file created: {requirements_file}")
        return str(requirements_file)
    
    def setup_deterministic_environment(self):
        """Set up environment for deterministic execution"""
        # Set random seeds
        import random
        import numpy as np
        
        seed = 42  # Fixed seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Set environment variables for deterministic behavior
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # For CUDA determinism
        
        # Try to set PyTorch deterministic mode
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            logger.warning("PyTorch not available for deterministic setup")
        
        logger.info(f"Deterministic environment set up with seed {seed}")
    
    def generate_reproducibility_report(self) -> str:
        """Generate a comprehensive reproducibility report"""
        env_info = self.get_environment_info()
        
        report = []
        report.append("=" * 60)
        report.append("REPRODUCIBILITY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append(f"Project Root: {self.project_root}")
        report.append("")
        
        # Environment info
        report.append("ENVIRONMENT INFORMATION:")
        report.append(f"  Python Version: {env_info.python_version}")
        report.append(f"  Platform: {env_info.platform}")
        report.append(f"  CUDA Available: {env_info.cuda_available}")
        if env_info.cuda_version:
            report.append(f"  CUDA Version: {env_info.cuda_version}")
        report.append(f"  GPU Count: {env_info.gpu_count}")
        report.append(f"  Memory: {env_info.memory_gb:.1f} GB")
        report.append(f"  CPU Count: {env_info.cpu_count}")
        report.append("")
        
        # Git info
        if env_info.git_sha:
            report.append("GIT INFORMATION:")
            report.append(f"  Commit: {env_info.git_sha}")
            report.append(f"  Branch: {env_info.git_branch}")
            report.append("")
        
        # Dependencies
        report.append("DEPENDENCIES:")
        report.append(f"  Total packages: {len(env_info.pip_freeze)}")
        for package in env_info.pip_freeze[:10]:  # Show first 10
            report.append(f"    {package}")
        if len(env_info.pip_freeze) > 10:
            report.append(f"    ... and {len(env_info.pip_freeze) - 10} more")
        report.append("")
        
        # Reproducibility hash
        report.append("REPRODUCIBILITY HASH:")
        report.append(f"  {self._calculate_reproducibility_hash(env_info)}")
        report.append("")
        
        return "\n".join(report)

def main():
    """Example usage of reproducibility validator"""
    validator = ReproducibilityValidator()
    
    # Create config
    config = ReproducibilityConfig(
        require_exact_python=True,
        require_cuda=False,
        min_memory_gb=4.0,
        require_git_clean=False,
        pin_dependencies=True,
        deterministic_seeds=True
    )
    
    # Validate environment
    validation = validator.validate_environment(config)
    print("Environment Validation:")
    print(f"  Passed: {validation['validation_passed']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    
    # Create reproducibility lock
    lock_data = validator.create_reproducibility_lock(config)
    print(f"\nReproducibility lock created with hash: {lock_data['reproducibility_hash']}")
    
    # Create requirements lock
    requirements_file = validator.create_requirements_lock()
    print(f"Requirements lock file created: {requirements_file}")
    
    # Generate report
    report = validator.generate_reproducibility_report()
    print(f"\n{report}")

if __name__ == "__main__":
    main()
