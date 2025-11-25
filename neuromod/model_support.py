"""
Model Support System for Neuromodulation

This module provides comprehensive model loading and management for both test
and production environments, with support for various model sizes and backends.

Key Features:
- Flexible model loading (test vs production)
- Memory management and quantization
- Multiple backend support (HuggingFace, vLLM)
- Model-specific attention hook paths
- Device mapping and optimization
"""

import os
import logging

# CRITICAL: Check CUDA_VISIBLE_DEVICES BEFORE importing torch
# Changing CUDA_VISIBLE_DEVICES after torch import causes CUDA initialization errors
_cuda_visible_devices_before_import = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cuda_visible_devices_before_import is not None:
    # Log this early so user knows what's set
    import sys
    print(f"⚠️  CUDA_VISIBLE_DEVICES is set to: {_cuda_visible_devices_before_import}", file=sys.stderr)
    if _cuda_visible_devices_before_import == '':
        print("⚠️  WARNING: CUDA_VISIBLE_DEVICES='' hides all GPUs from PyTorch!", file=sys.stderr)
        print("⚠️  To fix: unset CUDA_VISIBLE_DEVICES before importing torch", file=sys.stderr)

# Try to ensure PyTorch can find its bundled CUDA libraries
# This helps in containers where system CUDA libraries might conflict
try:
    # We'll set this after torch import to avoid circular dependency
    pass
except:
    pass

import torch

# After torch import, try to add PyTorch's bundled CUDA libraries to LD_LIBRARY_PATH
# This can help if system CUDA libraries are causing conflicts
try:
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(torch_lib_path):
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if torch_lib_path not in current_ld_path:
            # Add PyTorch's lib to the front of LD_LIBRARY_PATH
            os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}" if current_ld_path else torch_lib_path
            # Note: This won't help if CUDA was already initialized, but it might help for future imports
except Exception:
    pass  # Silently fail if we can't modify LD_LIBRARY_PATH
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import gc
import psutil
import platform
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if CUDA_VISIBLE_DEVICES was changed after torch import
_cuda_visible_devices_after_import = os.environ.get('CUDA_VISIBLE_DEVICES', None)
if _cuda_visible_devices_before_import != _cuda_visible_devices_after_import:
    logger.error("=" * 60)
    logger.error("CRITICAL: CUDA_VISIBLE_DEVICES was changed AFTER torch import!")
    logger.error("This will cause CUDA initialization to fail.")
    logger.error(f"Before import: {_cuda_visible_devices_before_import}")
    logger.error(f"After import: {_cuda_visible_devices_after_import}")
    logger.error("SOLUTION: Set CUDA_VISIBLE_DEVICES BEFORE importing torch or any modules that import torch")
    logger.error("=" * 60)

class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"        # < 1B parameters (test only)
    SMALL = "small"      # 1-7B parameters (test + limited production)
    MEDIUM = "medium"    # 7-30B parameters (production)
    LARGE = "large"      # 30-70B parameters (production)
    XLARGE = "xlarge"    # 70B+ parameters (production)

class BackendType(Enum):
    """Model backend types"""
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"
    MOCK = "mock"  # For testing without actual models

@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    name: str
    size: ModelSize
    backend: BackendType
    quantization: Optional[str] = None
    max_length: int = 2048
    device_map: Optional[str] = None
    torch_dtype: Optional[torch.dtype] = None
    trust_remote_code: bool = True
    low_cpu_mem_usage: bool = True
    test_mode: bool = False

@dataclass
class SystemInfo:
    """System resource information"""
    total_memory_gb: float
    available_memory_gb: float
    gpu_memory_gb: Optional[float] = None
    gpu_count: int = 0
    cpu_count: int = 0
    platform: str = "unknown"

class ModelSupportManager:
    """Manages model loading and resource allocation"""
    
    def __init__(self, test_mode: bool = True):
        self.test_mode = test_mode
        
        # Try to work around CUDA initialization issues
        self._try_fix_cuda_access()
        
        self.system_info = self._get_system_info()
        
        # Log system information after it's been initialized
        logger.info(f"ModelSupportManager initialized in {'TEST' if self.test_mode else 'PRODUCTION'} mode")
        logger.info(f"System: {self.system_info.platform}")
        logger.info(f"CPU Memory: {self.system_info.available_memory_gb:.1f} GB available / {self.system_info.total_memory_gb:.1f} GB total")
        if self.system_info.gpu_count > 0:
            logger.info(f"GPU: {self.system_info.gpu_count} device(s) detected")
            logger.info(f"GPU Memory: {self.system_info.gpu_memory_gb:.1f} GB per device")
            if torch.cuda.is_available():
                for i in range(self.system_info.gpu_count):
                    try:
                        logger.info(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                    except:
                        logger.info(f"  - GPU {i}: Available")
        else:
            logger.info("GPU: No GPU detected")
        
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs = self._define_model_configs()
    
    def _try_fix_cuda_access(self):
        """Try to work around CUDA initialization issues"""
        # Check if CUDA_VISIBLE_DEVICES might be causing issues
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        
        # If CUDA_VISIBLE_DEVICES is set but CUDA isn't available, warn
        try:
            cuda_available = torch.cuda.is_available()
            if not cuda_available and cuda_visible is not None:
                logger.warning("=" * 60)
                logger.warning("CUDA ACCESS ISSUE DETECTED")
                logger.warning(f"CUDA_VISIBLE_DEVICES={cuda_visible} but torch.cuda.is_available()=False")
                logger.warning("")
                logger.warning("This is often caused by:")
                logger.warning("  1. CUDA_VISIBLE_DEVICES being changed after torch import")
                logger.warning("  2. CUDA runtime initialization failure")
                logger.warning("  3. Multiple processes competing for CUDA")
                logger.warning("")
                logger.warning("WORKAROUND ATTEMPTS:")
                logger.warning("  1. Restart Python process (CUDA state cannot be reset)")
                logger.warning("  2. Unset CUDA_VISIBLE_DEVICES: unset CUDA_VISIBLE_DEVICES")
                logger.warning("  3. Set CUDA_VISIBLE_DEVICES before Python starts")
                logger.warning("  4. Check for other processes using CUDA: nvidia-smi")
                logger.warning("=" * 60)
        except Exception as e:
            logger.debug(f"Could not check CUDA availability: {e}")
    
    def _get_system_info(self) -> SystemInfo:
        """Get system resource information"""
        # Memory info
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # GPU info - try multiple methods to detect GPUs
        gpu_memory_gb = None
        gpu_count = 0
        
        # First, try PyTorch CUDA
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"CUDA availability check failed: {e}")
            logger.warning("This may be due to CUDA initialization issues. Checking alternative methods...")
        
        if cuda_available:
            try:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception as e:
                logger.warning(f"Failed to get CUDA device info: {e}")
        
        # If PyTorch CUDA failed, try nvidia-smi as fallback
        if gpu_count == 0:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=count,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    gpu_count = len(lines)
                    if gpu_count > 0:
                        # Get memory from first GPU
                        first_line = lines[0].split(',')
                        if len(first_line) >= 2:
                            try:
                                gpu_memory_gb = float(first_line[1].strip()) / 1024  # Convert MB to GB
                                logger.info(f"Detected {gpu_count} GPU(s) via nvidia-smi (PyTorch CUDA unavailable)")
                                logger.warning("CUDA is not available in PyTorch, but GPUs are detected via nvidia-smi")
                                logger.warning("This may indicate a CUDA/PyTorch compatibility issue")
                                logger.warning("Check CUDA_VISIBLE_DEVICES and ensure PyTorch was built with CUDA support")
                            except ValueError:
                                pass
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                logger.debug(f"nvidia-smi check failed: {e}")
        
        # Check CUDA_VISIBLE_DEVICES environment variable
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if cuda_visible_devices is not None:
            logger.info(f"CUDA_VISIBLE_DEVICES is set to: {cuda_visible_devices}")
            if cuda_visible_devices == '':
                logger.warning("CUDA_VISIBLE_DEVICES is set to empty string - this hides all GPUs from PyTorch!")
        
        # Check PyTorch CUDA build information
        pytorch_cuda_version = None
        pytorch_built_with_cuda = False
        try:
            pytorch_cuda_version = torch.version.cuda
            pytorch_built_with_cuda = pytorch_cuda_version is not None
        except:
            pass
        
        # Get system CUDA version from nvidia-smi if available
        system_cuda_version = None
        if gpu_count > 0 and not cuda_available:
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    system_cuda_version = result.stdout.strip().split('\n')[0]
            except:
                pass
        
        # Log CUDA diagnostic information
        if gpu_count > 0 and not cuda_available:
            logger.error("=" * 60)
            logger.error("🚨 CUDA INITIALIZATION FAILED - DIAGNOSTIC INFORMATION:")
            logger.error(f"PyTorch Version: {torch.__version__}")
            logger.error(f"PyTorch Built with CUDA: {pytorch_built_with_cuda}")
            if pytorch_cuda_version:
                logger.error(f"PyTorch CUDA Version: {pytorch_cuda_version}")
            else:
                logger.error("PyTorch CUDA Version: None (PyTorch may be CPU-only build)")
            if system_cuda_version:
                logger.error(f"System CUDA Driver Version: {system_cuda_version}")
            logger.error(f"CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
            
            # Check library paths
            ld_library_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
            library_path = os.environ.get('LIBRARY_PATH', 'Not set')
            logger.error(f"LD_LIBRARY_PATH: {ld_library_path}")
            logger.error(f"LIBRARY_PATH: {library_path}")
            
            # Try to check if CUDA libraries exist
            try:
                import subprocess
                cuda_lib_paths = [
                    '/usr/local/cuda/lib64',
                    '/usr/lib/x86_64-linux-gnu',
                    '/usr/local/cuda-12.8/lib64',
                    '/usr/local/cuda-12.1/lib64',
                ]
                if ld_library_path != 'Not set':
                    cuda_lib_paths.extend(ld_library_path.split(':'))
                
                # Also check PyTorch's bundled CUDA libraries
                try:
                    torch_path = os.path.dirname(torch.__file__)
                    torch_lib_path = os.path.join(torch_path, 'lib')
                    if os.path.exists(torch_lib_path):
                        cuda_lib_paths.append(torch_lib_path)
                except:
                    pass
                
                logger.error("Checking for CUDA libraries:")
                found_any = False
                for lib_path in set(cuda_lib_paths):
                    if lib_path and os.path.exists(lib_path):
                        try:
                            cuda_libs = [f for f in os.listdir(lib_path) if 'libcudart' in f or 'libcuda' in f]
                            if cuda_libs:
                                logger.error(f"  ✓ {lib_path}: Found {len(cuda_libs)} CUDA libraries")
                                found_any = True
                            else:
                                logger.error(f"  ✗ {lib_path}: No CUDA libraries found")
                        except PermissionError:
                            logger.error(f"  ? {lib_path}: Permission denied")
                        except Exception as e:
                            logger.debug(f"Error checking {lib_path}: {e}")
                
                if not found_any:
                    logger.error("  ⚠️  No CUDA libraries found in any checked paths!")
                    logger.error("  This suggests CUDA runtime libraries are missing.")
            except Exception as e:
                logger.debug(f"Could not check CUDA libraries: {e}")
            
            logger.error("=" * 60)
            
            # Provide specific recommendations based on the error
            logger.error("🔧 SOLUTIONS (try in order):")
            logger.error("")
            
            if not pytorch_built_with_cuda:
                logger.error("1. PyTorch was built WITHOUT CUDA support!")
                logger.error("   Fix: Install CUDA-enabled PyTorch:")
                logger.error("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                logger.error("   Then RESTART Python")
            else:
                logger.error("1. CUDA Library Path Issue (COMMON IN CONTAINERS):")
                logger.error("   PyTorch may not be finding CUDA runtime libraries.")
                logger.error("   ")
                logger.error("   Fix:")
                logger.error("     a) Check if CUDA libraries are in LD_LIBRARY_PATH:")
                logger.error("        echo $LD_LIBRARY_PATH")
                logger.error("        ls -la /usr/local/cuda/lib64/ | grep libcudart")
                logger.error("     b) If missing, add to LD_LIBRARY_PATH:")
                logger.error("        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
                logger.error("     c) RESTART Python after setting LD_LIBRARY_PATH")
                logger.error("")
                logger.error("2. CUDA_VISIBLE_DEVICES Issue:")
                logger.error("   The error 'CUDA unknown error' can mean CUDA_VISIBLE_DEVICES")
                logger.error("   was changed AFTER torch was imported.")
                logger.error("   ")
                logger.error("   Fix:")
                logger.error("     a) RESTART Python (CUDA state cannot be reset)")
                logger.error("     b) Set CUDA_VISIBLE_DEVICES BEFORE starting Python:")
                logger.error("        export CUDA_VISIBLE_DEVICES=0,1,2,3  # or unset it")
                logger.error("        python your_script.py")
                logger.error("")
                logger.error("3. PyTorch Installation Issue:")
                logger.error("   Verify PyTorch CUDA installation:")
                logger.error("     python -c 'import torch; print(torch.version.cuda)'")
                logger.error("     # Should show CUDA version (e.g., 12.1)")
                logger.error("     # If None, reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu121")
                logger.error("")
                logger.error("4. Container/Docker Issue (LIKELY CAUSE):")
                logger.error("     The 'CUDA unknown error' in containers often means:")
                logger.error("     a) Container not started with proper GPU access:")
                logger.error("        docker run --gpus all ...")
                logger.error("        # OR for docker-compose:")
                logger.error("        deploy:")
                logger.error("          resources:")
                logger.error("            reservations:")
                logger.error("              devices:")
                logger.error("                - driver: nvidia")
                logger.error("                  capabilities: [gpu]")
                logger.error("     b) NVIDIA Container Toolkit not installed on host")
                logger.error("     c) CUDA runtime mismatch - PyTorch bundles CUDA 12.1")
                logger.error("        but container has CUDA 12.8 - try setting:")
                logger.error("        export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
                logger.error("     d) Try using PyTorch's bundled CUDA libraries:")
                logger.error("        # PyTorch should use its own CUDA libs, but if not:")
                logger.error("        python -c 'import torch; print(torch.__file__)'")
                logger.error("        # Then add torch/lib to LD_LIBRARY_PATH")
                logger.error("     e) RESTART the container completely")
                logger.error("")
                logger.error("5. Verify Container GPU Access:")
                logger.error("     # In container, check:")
                logger.error("     nvidia-smi  # Should show GPUs")
                logger.error("     ls -la /dev/nvidia*  # Should show NVIDIA device files")
                logger.error("     # If /dev/nvidia* doesn't exist, container lacks GPU access")
                logger.error("")
                logger.error("6. Container Restart Required:")
                logger.error("     # CUDA state cannot be reset - you MUST restart the container")
                logger.error("     # Before restarting, ensure container is started with:")
                logger.error("     docker run --gpus all --runtime=nvidia ...")
                logger.error("     # OR set environment in docker-compose:")
                logger.error("     environment:")
                logger.error("       - NVIDIA_VISIBLE_DEVICES=all")
                logger.error("       - NVIDIA_DRIVER_CAPABILITIES=compute,utility")
            logger.error("=" * 60)
        
        return SystemInfo(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            gpu_memory_gb=gpu_memory_gb,
            gpu_count=gpu_count,
            cpu_count=psutil.cpu_count(),
            platform=platform.system()
        )
    
    def _define_model_configs(self) -> Dict[str, ModelConfig]:
        """Define model configurations for different environments"""
        configs = {}
        
        if self.test_mode:
            # Test environment - small models only
            configs.update({
                "gpt2": ModelConfig(
                    name="gpt2",
                    size=ModelSize.TINY,
                    backend=BackendType.HUGGINGFACE,
                    max_length=512,
                    test_mode=True
                ),
                "distilgpt2": ModelConfig(
                    name="distilgpt2", 
                    size=ModelSize.TINY,
                    backend=BackendType.HUGGINGFACE,
                    max_length=512,
                    test_mode=True
                ),
                "microsoft/DialoGPT-small": ModelConfig(
                    name="microsoft/DialoGPT-small",
                    size=ModelSize.TINY,
                    backend=BackendType.HUGGINGFACE,
                    max_length=1024,
                    test_mode=True
                ),
                "mock": ModelConfig(
                    name="mock",
                    size=ModelSize.TINY,
                    backend=BackendType.MOCK,
                    max_length=512,
                    test_mode=True
                ),
                "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
                    name="meta-llama/Llama-3.1-8B-Instruct",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16,
                    test_mode=True
                ),
                "meta-llama/Llama-3.1-8B": ModelConfig(
                    name="meta-llama/Llama-3.1-8B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16,
                    test_mode=True
                )
            })
        else:
            # Production environment - full model suite
            configs.update({
                "meta-llama/Llama-3.1-8B-Instruct": ModelConfig(
                    name="meta-llama/Llama-3.1-8B-Instruct",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-3.1-8B": ModelConfig(
                    name="meta-llama/Llama-3.1-8B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
                    name="meta-llama/Llama-3.1-70B-Instruct",
                    size=ModelSize.LARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization=None,
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-3.1-70B": ModelConfig(
                    name="meta-llama/Llama-3.1-70B",
                    size=ModelSize.LARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization=None,
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Meta-Llama-3.1-70B-Instruct": ModelConfig(
                    name="meta-llama/Meta-Llama-3.1-70B-Instruct",
                    size=ModelSize.LARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization=None,
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-4-Scout-17B-16E-Instruct": ModelConfig(
                    name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    size=ModelSize.MEDIUM,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-4-Maverick-17B-128E": ModelConfig(
                    name="meta-llama/Llama-4-Maverick-17B-128E",
                    size=ModelSize.MEDIUM,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-4-Maverick-17B-128E-Instruct": ModelConfig(
                    name="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                    size=ModelSize.MEDIUM,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "Qwen/Qwen-2.5-7B": ModelConfig(
                    name="Qwen/Qwen-2.5-7B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "Qwen/Qwen2.5-Omni-7B": ModelConfig(
                    name="Qwen/Qwen2.5-Omni-7B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "Qwen/Qwen-2.5-Omni-7B": ModelConfig(
                    name="Qwen/Qwen2.5-Omni-7B",  # Alias for correct name
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "Qwen/Qwen2.5-32B-Instruct": ModelConfig(
                    name="Qwen/Qwen2.5-32B-Instruct",
                    size=ModelSize.MEDIUM,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=32768,
                    torch_dtype=torch.float16
                ),
                "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelConfig(
                    name="mistralai/Mixtral-8x22B-Instruct-v0.1",
                    size=ModelSize.XLARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "openai/gpt-oss-20b": ModelConfig(
                    name="openai/gpt-oss-20b",
                    size=ModelSize.MEDIUM,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=4096,
                    torch_dtype=torch.bfloat16  # GPT-OSS models with Mxfp4Config use bfloat16
                ),
                "openai/gpt-oss-120b": ModelConfig(
                    name="openai/gpt-oss-120b",
                    size=ModelSize.XLARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=4096,
                    torch_dtype=torch.bfloat16  # GPT-OSS models with Mxfp4Config use bfloat16
                )
            })
        
        return configs
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for current environment"""
        return list(self.model_configs.keys())
    
    def get_recommended_model(self) -> str:
        """Get recommended model for current environment"""
        if self.test_mode:
            return "gpt2"  # Smallest model for testing
        else:
            # Recommend based on available resources
            if self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 80:
                # Full 120B model support assumes multi-GPU or very high VRAM
                return "openai/gpt-oss-120b"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 48:
                return "openai/gpt-oss-20b"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 40:
                return "meta-llama/Llama-3.1-70B-Instruct"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 24:
                # 32B models with 4-bit quantization need ~20-24GB VRAM
                return "Qwen/Qwen2.5-32B-Instruct"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 18:
                # 17B models with 4-bit quantization need ~16-18GB VRAM
                return "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 16:
                return "meta-llama/Llama-3.1-8B-Instruct"
            else:
                return "Qwen/Qwen-2.5-7B"
    
    def load_model(self, model_name: str, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load a model with appropriate configuration
        
        Returns:
            Tuple of (model, tokenizer, model_info)
        """
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not available. Available: {self.get_available_models()}")
        
        config = self.model_configs[model_name]
        
        # Check if already loaded
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        # Check resource requirements
        if not self._check_resource_requirements(config):
            raise RuntimeError(f"Insufficient resources for model {model_name}")
        
        logger.info(f"Loading model: {model_name}")
        
        try:
            if config.backend == BackendType.MOCK:
                model, tokenizer, model_info = self._load_mock_model(config)
            else:
                model, tokenizer, model_info = self._load_huggingface_model(config, **kwargs)
            
            # Store loaded model
            self.loaded_models[model_name] = (model, tokenizer, model_info)
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer, model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _load_mock_model(self, config: ModelConfig) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load a mock model for testing"""
        from transformers import AutoTokenizer
        
        # Load tokenizer for consistency
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create mock model
        class MockModel:
            def __init__(self):
                self.config = type('Config', (), {
                    'vocab_size': tokenizer.vocab_size,
                    'hidden_size': 768,
                    'num_attention_heads': 12,
                    'num_hidden_layers': 12
                })()
            
            def generate(self, input_ids, **kwargs):
                # Generate random tokens for testing
                batch_size, seq_len = input_ids.shape
                vocab_size = self.config.vocab_size
                
                # Simple random generation
                new_tokens = torch.randint(0, vocab_size, (batch_size, kwargs.get('max_new_tokens', 50)))
                return torch.cat([input_ids, new_tokens], dim=1)
            
            def __call__(self, input_ids, **kwargs):
                # Return mock logits
                batch_size, seq_len = input_ids.shape
                vocab_size = self.config.vocab_size
                logits = torch.randn(batch_size, seq_len, vocab_size)
                return type('Output', (), {'logits': logits})()
        
        model = MockModel()
        model_info = {
            "name": config.name,
            "size": config.size.value,
            "backend": config.backend.value,
            "test_mode": True,
            "parameters": "mock"
        }
        
        return model, tokenizer, model_info
    
    def _load_huggingface_model(self, config: ModelConfig, **kwargs) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load a HuggingFace model"""
        # Disable accelerate to avoid MPS issues on older PyTorch versions
        if not hasattr(torch, 'mps') or not torch.backends.mps.is_available():
            os.environ['ACCELERATE_DISABLE_RICH'] = '1'
            os.environ['ACCELERATE_USE_CPU'] = '1'
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Get HuggingFace token from environment or kwargs
        hf_token = kwargs.get('token') or os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        
        # Check if this is a Meta Llama model that requires authentication
        requires_auth = 'meta-llama' in config.name.lower()
        if requires_auth and not hf_token:
            logger.warning(
                f"⚠️  Model {config.name} requires HuggingFace authentication.\n"
                f"   Please set HUGGINGFACE_HUB_TOKEN environment variable or accept the license at:\n"
                f"   https://huggingface.co/{config.name}\n"
                f"   Then run: huggingface-cli login"
            )
        
        # Load tokenizer with token if available (with retry for network issues)
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.name,
                    token=hf_token,
                    resume_download=True  # Enable resume for interrupted downloads
                )
                break  # Success
            except Exception as e:
                error_str = str(e).lower()
                is_network_error = (
                    'incompleteread' in error_str or
                    'connection broken' in error_str or
                    'connection error' in error_str or
                    'timeout' in error_str
                )
                
                if is_network_error and attempt < max_retries - 1:
                    logger.warning(f"Network error loading tokenizer (attempt {attempt + 1}/{max_retries}), retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    # Check for authentication errors
                    if '401' in error_str or 'authentication' in error_str or 'permission' in error_str:
                        raise RuntimeError(
                            f"❌ Authentication required for {config.name}\n"
                            f"   1. Accept the license at: https://huggingface.co/{config.name}\n"
                            f"   2. Get your token from: https://huggingface.co/settings/tokens\n"
                            f"   3. Set environment variable: export HUGGINGFACE_HUB_TOKEN=your_token\n"
                            f"   4. Or run: huggingface-cli login\n"
                            f"   Original error: {e}"
                        )
                    raise
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine device and dtype
        device_map = self._get_device_map(config)
        torch_dtype = config.torch_dtype or torch.float32
        
        # Check if model is pre-quantized (e.g., GPT-OSS models with Mxfp4Config)
        is_pre_quantized = False
        model_config = None
        try:
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(
                config.name,
                token=hf_token,
                trust_remote_code=config.trust_remote_code
            )
            # Check if model config indicates pre-quantization
            if hasattr(model_config, 'quantization_config') and model_config.quantization_config is not None:
                is_pre_quantized = True
                logger.info(f"Model {config.name} is pre-quantized, skipping BitsAndBytesConfig")
            # Also check for GPT-OSS models specifically (they use Mxfp4Config)
            if 'gpt-oss' in config.name.lower():
                is_pre_quantized = True
                logger.info(f"GPT-OSS model detected, skipping BitsAndBytesConfig (uses Mxfp4Config)")
        except Exception as e:
            logger.debug(f"Could not check model config for pre-quantization: {e}")
        
        # Determine CUDA availability early
        cuda_available = False
        try:
            cuda_available = torch.cuda.is_available()
        except Exception as e:
            logger.debug(f"CUDA availability check failed: {e}")
            cuda_available = False
        
        # Setup quantization if specified and model is not pre-quantized
        quantization_config = None
        enable_cpu_offload = False
        
        # NEVER enable CPU offloading when GPU is available - it causes device mismatch issues
        # Only use CPU if there's no GPU available
        if config.quantization and not self.test_mode and not is_pre_quantized:
            try:
                from transformers import BitsAndBytesConfig
                
                # Only enable CPU offloading if there's NO GPU available
                # When GPU is available, we'll use GPU-only loading even if it means OOM errors
                # (User can use smaller models or reduce quantization if needed)
                if not cuda_available or self.system_info.gpu_count == 0:
                    enable_cpu_offload = True
                    logger.info("No GPU available, enabling CPU offloading for quantized model")
                else:
                    enable_cpu_offload = False
                    logger.info("GPU available - disabling CPU offloading to prevent device mismatch issues")
                
                if config.quantization == "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True
                    )
                    logger.info(f"Using 4-bit quantization for {config.name}")
                elif config.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch_dtype,
                        llm_int8_enable_fp32_cpu_offload=enable_cpu_offload
                    )
                    if enable_cpu_offload:
                        logger.info(f"Using 8-bit quantization with CPU offloading for {config.name} (no GPU)")
                    else:
                        logger.info(f"Using 8-bit quantization for {config.name} (GPU-only, no CPU offloading)")
            except ImportError:
                logger.warning("bitsandbytes not available, loading model without quantization")
                quantization_config = None
        elif is_pre_quantized:
            logger.info(f"Skipping quantization config for pre-quantized model {config.name}")
        
        # Check if MPS is available to avoid accelerate issues on Mac
        # Only force CPU loading if we're on Mac with MPS (to avoid accelerate issues)
        # Otherwise, use GPU if CUDA is available
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # Force CPU loading to avoid accelerate MPS issues on Mac
            device_map = None  # Don't use device_map at all
            kwargs.pop('device_map', None)  # Remove device_map from kwargs
            # Override low_cpu_mem_usage if it exists
            kwargs.pop('low_cpu_mem_usage', None)
            kwargs['low_cpu_mem_usage'] = False
            logger.info("=" * 60)
            logger.info("DEVICE MODE: CPU (MPS detected on Mac - forcing CPU to avoid accelerate issues)")
            logger.info(f"CPU Memory Available: {self.system_info.available_memory_gb:.2f} GB / {self.system_info.total_memory_gb:.2f} GB")
            logger.info("=" * 60)
        elif device_map == "cpu" or (device_map is None and self.test_mode and config.size == ModelSize.TINY):
            # Test mode with TINY models or explicit CPU mapping - use CPU
            logger.info("=" * 60)
            if self.test_mode and config.size == ModelSize.TINY:
                logger.info("DEVICE MODE: CPU (Test mode - using CPU for tiny models)")
            elif device_map == "cpu":
                logger.info("DEVICE MODE: CPU (Explicit CPU mapping)")
            else:
                logger.info("DEVICE MODE: CPU")
            logger.info(f"CPU Memory Available: {self.system_info.available_memory_gb:.2f} GB / {self.system_info.total_memory_gb:.2f} GB")
            logger.info("=" * 60)
        elif cuda_available:
            # CUDA is available - use specific GPU device instead of "auto" to prevent CPU offloading
            if device_map is None:
                device_map = "cuda:0"  # Use specific GPU device to prevent CPU offloading
            
            # Get GPU memory info
            gpu_memory_total = self.system_info.gpu_memory_gb or 0.0
            gpu_memory_allocated = 0.0
            gpu_memory_reserved = 0.0
            gpu_memory_free = gpu_memory_total
            
            try:
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    gpu_memory_free = gpu_memory_total - gpu_memory_reserved
            except:
                pass
            
            logger.info("=" * 60)
            logger.info(f"DEVICE MODE: GPU (CUDA)")
            try:
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown'
            except:
                gpu_name = 'Unknown'
            logger.info(f"GPU Device: {gpu_name}")
            logger.info(f"GPU Memory Total: {gpu_memory_total:.2f} GB")
            logger.info(f"GPU Memory Free: {gpu_memory_free:.2f} GB")
            logger.info(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
            logger.info(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
            logger.info(f"Device Map: {device_map}")
            logger.info("=" * 60)
        elif self.system_info.gpu_count > 0:
            # GPUs detected via nvidia-smi but PyTorch CUDA not available
            logger.warning("=" * 60)
            logger.warning("⚠️  GPUs DETECTED BUT CUDA UNAVAILABLE IN PYTORCH")
            logger.warning(f"Detected {self.system_info.gpu_count} GPU(s) via nvidia-smi")
            logger.warning(f"GPU Memory: {self.system_info.gpu_memory_gb:.2f} GB per device")
            logger.warning("PyTorch cannot access CUDA - this may be due to:")
            logger.warning("  1. CUDA_VISIBLE_DEVICES environment variable issues")
            logger.warning("  2. PyTorch built without CUDA support")
            logger.warning("  3. CUDA driver/runtime version mismatch")
            logger.warning("  4. CUDA initialization error (check earlier warnings)")
            logger.warning("")
            logger.warning("Falling back to CPU mode. Model will be dequantized.")
            logger.warning("")
            logger.warning("QUICK FIX ATTEMPTS:")
            logger.warning("  1. Unset CUDA_VISIBLE_DEVICES and restart Python:")
            logger.warning("     unset CUDA_VISIBLE_DEVICES")
            logger.warning("  2. Check PyTorch CUDA build:")
            logger.warning("     python -c 'import torch; print(torch.version.cuda)'")
            logger.warning("  3. If None, reinstall PyTorch with CUDA:")
            logger.warning("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
            logger.warning("=" * 60)
            device_map = None
        else:
            # No GPU available - using CPU
            logger.info("=" * 60)
            logger.info("DEVICE MODE: CPU (No GPU available)")
            logger.info(f"CPU Memory Available: {self.system_info.available_memory_gb:.2f} GB / {self.system_info.total_memory_gb:.2f} GB")
            logger.info("=" * 60)
        
        # Load model
        load_kwargs = {
            'trust_remote_code': config.trust_remote_code,
            'low_cpu_mem_usage': kwargs.get('low_cpu_mem_usage', config.low_cpu_mem_usage),
            'use_safetensors': True,  # Prefer safetensors format
            'attn_implementation': 'eager',  # FIX: Force eager attention to support output_attentions=True for induction head detection
            **{k: v for k, v in kwargs.items() if k not in ['low_cpu_mem_usage', 'device_map', 'use_safetensors', 'torch_dtype', 'dtype', 'attn_implementation']}
        }
        
        # Only set dtype if model is not pre-quantized
        # Pre-quantized models (like GPT-OSS with Mxfp4Config) have their own dtype
        if not is_pre_quantized:
            load_kwargs['dtype'] = torch_dtype
        else:
            # For pre-quantized models, use the dtype from config but let the model decide
            # GPT-OSS models with Mxfp4Config use bfloat16 internally
            if 'gpt-oss' in config.name.lower():
                load_kwargs['dtype'] = torch.bfloat16
                logger.info(f"Using bfloat16 for GPT-OSS model {config.name} (Mxfp4Config requires bfloat16)")
            elif model_config is not None:
                # For other pre-quantized models, try to detect from config
                try:
                    if hasattr(model_config, 'torch_dtype') and model_config.torch_dtype is not None:
                        dtype_str = str(model_config.torch_dtype).replace('torch.', '')
                        if hasattr(torch, dtype_str):
                            load_kwargs['dtype'] = getattr(torch, dtype_str)
                        else:
                            load_kwargs['dtype'] = torch_dtype
                    else:
                        load_kwargs['dtype'] = torch_dtype
                except:
                    load_kwargs['dtype'] = torch_dtype
            else:
                load_kwargs['dtype'] = torch_dtype
        
        # Add quantization config if available
        if quantization_config is not None:
            load_kwargs['quantization_config'] = quantization_config
        
        # Only add device_map if it's not None
        # When GPU is available, use specific GPU device instead of "auto" to prevent CPU offloading
        if device_map is not None:
            if enable_cpu_offload and quantization_config is not None:
                # For CPU offloading (no GPU), use "auto" which allows CPU/GPU distribution
                if device_map == "auto":
                    load_kwargs['device_map'] = "auto"
                else:
                    load_kwargs['device_map'] = device_map
                    logger.info("Using custom device_map with CPU offloading enabled")
            else:
                # When GPU is available, use specific GPU device instead of "auto"
                # This prevents HuggingFace from offloading to CPU
                if device_map == "auto" and cuda_available and self.system_info.gpu_count > 0:
                    # Use specific GPU device (cuda:0) instead of "auto" to prevent CPU offloading
                    load_kwargs['device_map'] = "cuda:0"
                    logger.info("Using device_map='cuda:0' instead of 'auto' to prevent CPU offloading")
                else:
                    load_kwargs['device_map'] = device_map
        elif enable_cpu_offload and quantization_config is not None:
            # If CPU offloading is needed but no device_map set, use "auto"
            load_kwargs['device_map'] = "auto"
            logger.info("Setting device_map='auto' to enable CPU offloading for quantized model")
        elif cuda_available and self.system_info.gpu_count > 0 and quantization_config is not None:
            # When GPU is available and using quantization, use specific GPU device
            load_kwargs['device_map'] = "cuda:0"
            logger.info("Setting device_map='cuda:0' for quantized model to prevent CPU offloading")
        
        # Add token if available
        if hf_token:
            load_kwargs['token'] = hf_token
        
        # Load model with error handling for authentication and network issues
        max_retries = 3
        retry_delay = 5  # seconds
        cpu_offload_retry = False
        
        for attempt in range(max_retries):
            try:
                # Special handling for Qwen2.5-Omni models (multimodal)
                if 'qwen2.5-omni' in config.name.lower() or 'qwen2_5_omni' in config.name.lower():
                    try:
                        from transformers import Qwen2_5OmniForConditionalGeneration
                        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                            config.name,
                            **load_kwargs,
                            resume_download=True  # Enable resume for interrupted downloads
                        )
                        logger.info(f"Loaded Qwen2.5-Omni model using Qwen2_5OmniForConditionalGeneration")
                    except ImportError:
                        raise RuntimeError(
                            f"Qwen2.5-Omni models require transformers>=4.51.3 with Qwen2.5-Omni support. "
                            f"Install with: pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview"
                        )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        config.name,
                        **load_kwargs,
                        resume_download=True  # Enable resume for interrupted downloads
                    )
                
                # Log where the model was actually loaded
                try:
                    if hasattr(model, 'hf_device_map'):
                        # Model was loaded with device_map
                        logger.info("=" * 60)
                        logger.info("MODEL LOADED SUCCESSFULLY")
                        logger.info(f"Model: {config.name}")
                        if model.hf_device_map:
                            logger.info(f"Device Map: {model.hf_device_map}")
                            # Count devices used
                            devices_used = set()
                            for layer_name, device in model.hf_device_map.items():
                                if isinstance(device, (int, str)):
                                    devices_used.add(str(device))
                            logger.info(f"Devices Used: {', '.join(sorted(devices_used))}")
                        else:
                            logger.info("Device Map: Not available (model may be on CPU)")
                    else:
                        # Check first parameter's device
                        first_param = next(model.parameters(), None)
                        if first_param is not None:
                            device = first_param.device
                            logger.info("=" * 60)
                            logger.info("MODEL LOADED SUCCESSFULLY")
                            logger.info(f"Model: {config.name}")
                            logger.info(f"Device: {device}")
                            if device.type == 'cuda':
                                logger.info(f"GPU Memory After Load: {torch.cuda.memory_allocated(device.index) / (1024**3):.2f} GB allocated")
                        else:
                            logger.info(f"Model {config.name} loaded successfully")
                    logger.info("=" * 60)
                except Exception as log_error:
                    logger.debug(f"Could not log device info: {log_error}")
                    logger.info(f"Model {config.name} loaded successfully")
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_str = str(e).lower()
                error_msg = str(e)
                
                # Check if it's a CPU/disk offloading error for quantized models
                is_offload_error = (
                    'some modules are dispatched on the cpu' in error_str or
                    'some modules are dispatched on the disk' in error_str or
                    'llm_int8_enable_fp32_cpu_offload' in error_str or
                    'not enough gpu ram' in error_str
                )
                
                # NEVER retry with CPU offloading when GPU is available - it causes device mismatch
                # Only retry with CPU offloading if there's truly no GPU
                if is_offload_error and quantization_config is not None and not enable_cpu_offload and not cpu_offload_retry:
                    if not cuda_available or self.system_info.gpu_count == 0:
                        # Only enable CPU offloading if there's no GPU
                        logger.warning("Quantized model doesn't fit in memory and no GPU available, enabling CPU offloading...")
                        cpu_offload_retry = True
                        
                        # Enable CPU offloading in quantization config
                        if hasattr(quantization_config, 'llm_int8_enable_fp32_cpu_offload'):
                            quantization_config.llm_int8_enable_fp32_cpu_offload = True
                        # Update load_kwargs
                        load_kwargs['quantization_config'] = quantization_config
                        if 'device_map' not in load_kwargs or load_kwargs.get('device_map') is None:
                            load_kwargs['device_map'] = "auto"
                        logger.info("Retrying with CPU offloading enabled...")
                        continue
                    else:
                        # GPU is available but model doesn't fit - don't enable CPU offloading
                        # This will cause an error, but that's better than device mismatch issues
                        logger.error("Model doesn't fit in GPU memory, but CPU offloading is disabled to prevent device mismatch.")
                        logger.error("Consider using a smaller model or reducing quantization level.")
                        raise
                
                # Check if it's a network/download error
                is_network_error = (
                    'incompleteread' in error_str or
                    'connection broken' in error_str or
                    'connection error' in error_str or
                    'timeout' in error_str or
                    'network' in error_str or
                    'download' in error_str or
                    'incompleteread' in error_msg.lower()
                )
                
                if is_network_error and attempt < max_retries - 1:
                    logger.warning(
                        f"Network error during download (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds... (download will resume from where it stopped)")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    # Not a network error, or out of retries - check for auth errors
                    error_str = str(e).lower()
                    error_msg = str(e)
                    
                    # Check for authentication or access issues
                    is_auth_error = (
                        '401' in error_msg or 
                        'authentication' in error_str or 
                        'permission' in error_str or 
                        'not found' in error_str or
                        'does not appear to have a file' in error_str or
                        'repository not found' in error_str
                    )
                    
                    if is_auth_error:
                        # Check if it's a Meta Llama model
                        if 'meta-llama' in config.name.lower():
                            raise RuntimeError(
                                f"❌ Failed to load {config.name}\n"
                                f"   This model requires HuggingFace authentication:\n"
                                f"   1. Accept the license at: https://huggingface.co/{config.name}\n"
                                f"      (Click 'Agree and access repository')\n"
                                f"   2. Get your access token:\n"
                                f"      - Visit: https://huggingface.co/settings/tokens\n"
                                f"      - Create a new token with 'Read' access\n"
                                f"   3. Set the token (Windows PowerShell):\n"
                                f"      $env:HUGGINGFACE_HUB_TOKEN='your_token_here'\n"
                                f"   4. Or use CLI: huggingface-cli login\n"
                                f"   \n"
                                f"   Original error: {error_msg}"
                            )
                        else:
                            raise RuntimeError(
                                f"❌ Failed to load {config.name}\n"
                                f"   Error: {error_msg}\n"
                                f"   \n"
                                f"   If this is a private model, ensure you have:\n"
                                f"   1. Access to the model repository\n"
                                f"   2. Set HUGGINGFACE_HUB_TOKEN environment variable\n"
                                f"   3. Run: huggingface-cli login"
                            )
                    # If not auth error, re-raise the original exception
                    raise
        
        # Get model info
        model_info = {
            "name": config.name,
            "size": config.size.value,
            "backend": config.backend.value,
            "device_map": device_map,
            "torch_dtype": str(torch_dtype),
            "parameters": self._estimate_parameters(model)
        }
        
        return model, tokenizer, model_info
    
    def _get_device_map(self, config: ModelConfig) -> str:
        """Determine optimal device mapping"""
        # In test mode, use CPU only for TINY models (truly fast tests)
        # For SMALL+ models, use GPU even in test mode (CPU is actually slower)
        if self.test_mode and config.size == ModelSize.TINY:
            return "cpu"  # Use CPU for tiny models in test mode
        
        if self.system_info.gpu_count == 0:
            return "cpu"
        
        # For production or larger test models, use specific GPU device instead of "auto"
        # This prevents HuggingFace from offloading to CPU
        return "cuda:0"  # Use specific GPU device to prevent CPU offloading
    
    def _check_resource_requirements(self, config: ModelConfig) -> bool:
        """Check if system has sufficient resources for model"""
        if self.test_mode:
            return True  # Test mode always allows loading
        
        # Estimate memory requirements (rough)
        memory_requirements = {
            ModelSize.TINY: 1.0,
            ModelSize.SMALL: 8.0,
            ModelSize.MEDIUM: 20.0,
            ModelSize.LARGE: 40.0,
            ModelSize.XLARGE: 80.0
        }
        
        required_memory = memory_requirements.get(config.size, 100.0)
        
        if config.quantization == "4bit":
            required_memory *= 0.25
        elif config.quantization == "8bit":
            required_memory *= 0.5
        
        available_memory = self.system_info.gpu_memory_gb or self.system_info.available_memory_gb
        
        return available_memory >= required_memory
    
    def _estimate_parameters(self, model) -> str:
        """Estimate number of parameters in model"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            if total_params >= 1e9:
                return f"{total_params/1e9:.1f}B"
            elif total_params >= 1e6:
                return f"{total_params/1e6:.1f}M"
            else:
                return f"{total_params:,}"
        except:
            return "unknown"
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a loaded model"""
        if model_name in self.loaded_models:
            _, _, model_info = self.loaded_models[model_name]
            return model_info
        return None
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models"""
        return list(self.loaded_models.keys())
    
    def cleanup(self):
        """Clean up all loaded models"""
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        logger.info("Cleaned up all models")

def create_model_support(test_mode: bool = True) -> ModelSupportManager:
    """Create a model support manager"""
    return ModelSupportManager(test_mode=test_mode)

def get_attention_hook_paths(model_name: str) -> Dict[str, str]:
    """Get attention hook paths for different model architectures"""
    hook_paths = {
        "gpt2": "transformer.h.{}.attn",
        "distilgpt2": "transformer.h.{}.attn",
        "meta-llama/Llama-3.1-8B": "model.layers.{}.self_attn",
        "meta-llama/Llama-3.1-8B-Instruct": "model.layers.{}.self_attn",
        "meta-llama/Llama-3.1-70B": "model.layers.{}.self_attn",
        "meta-llama/Llama-3.1-70B-Instruct": "model.layers.{}.self_attn",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "model.layers.{}.self_attn",
        "meta-llama/Meta-Llama-3-70B": "model.layers.{}.self_attn",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct": "model.layers.{}.self_attn",
        "Qwen/Qwen-2.5-7B": "model.layers.{}.self_attn",
        "Qwen/Qwen2.5-32B": "model.layers.{}.self_attn",
        "Qwen/Qwen2.5-Omni-7B": "model.layers.{}.self_attn",
        "Qwen/Qwen-2.5-Omni-7B": "model.layers.{}.self_attn",  # Alias
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "model.layers.{}.self_attn",
        "openai/gpt-oss-20b": "model.layers.{}.self_attn",
        "openai/gpt-oss-120b": "model.layers.{}.self_attn"
    }
    
    # Find matching pattern (check for both base name and full name)
    for pattern, path in hook_paths.items():
        if pattern in model_name or model_name in pattern:
            return {
                "attention": path,
                "mlp": path.replace("self_attn", "mlp"),
                "layer_norm": path.replace("self_attn", "input_layernorm")
            }
    
    # Default fallback
    return {
        "attention": "model.layers.{}.self_attn",
        "mlp": "model.layers.{}.mlp", 
        "layer_norm": "model.layers.{}.input_layernorm"
    }

def main():
    """Example usage of the model support system"""
    print("🧠 Model Support System Demo")
    print("=" * 50)
    
    # Create model support manager
    manager = create_model_support(test_mode=True)
    
    print(f"Available models: {manager.get_available_models()}")
    print(f"Recommended model: {manager.get_recommended_model()}")
    print(f"System info: {manager.system_info}")
    
    # Load a test model
    try:
        model_name = manager.get_recommended_model()
        print(f"\nLoading model: {model_name}")
        
        model, tokenizer, model_info = manager.load_model(model_name)
        print(f"Model loaded successfully!")
        print(f"Model info: {model_info}")
        
        # Test generation
        prompt = "Hello, how are you?"
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        
        # Cleanup
        manager.cleanup()
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
