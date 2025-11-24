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

import torch
import logging
import os
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
        self.system_info = self._get_system_info()
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs = self._define_model_configs()
        
        logger.info(f"ModelSupportManager initialized in {'TEST' if test_mode else 'PRODUCTION'} mode")
        logger.info(f"System: {self.system_info.platform}, Memory: {self.system_info.available_memory_gb:.1f}GB")
    
    def _get_system_info(self) -> SystemInfo:
        """Get system resource information"""
        # Memory info
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # GPU info
        gpu_memory_gb = None
        gpu_count = 0
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
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
                "meta-llama/Llama-3.1-70B-Instruct": ModelConfig(
                    name="meta-llama/Llama-3.1-70B-Instruct",
                    size=ModelSize.LARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
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
                    torch_dtype=torch.float16
                ),
                "openai/gpt-oss-120b": ModelConfig(
                    name="openai/gpt-oss-120b",
                    size=ModelSize.XLARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=4096,
                    torch_dtype=torch.float16
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
        
        # Setup quantization if specified and model is not pre-quantized
        quantization_config = None
        if config.quantization and not self.test_mode and not is_pre_quantized:
            try:
                from transformers import BitsAndBytesConfig
                
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
                        bnb_8bit_compute_dtype=torch_dtype
                    )
                    logger.info(f"Using 8-bit quantization for {config.name}")
            except ImportError:
                logger.warning("bitsandbytes not available, loading model without quantization")
                quantization_config = None
        elif is_pre_quantized:
            logger.info(f"Skipping quantization config for pre-quantized model {config.name}")
        
        # Check if MPS is available to avoid accelerate issues
        if not hasattr(torch, 'mps') or not torch.backends.mps.is_available():
            # Force CPU loading to avoid accelerate MPS issues
            device_map = None  # Don't use device_map at all
            kwargs.pop('device_map', None)  # Remove device_map from kwargs
            # Override low_cpu_mem_usage if it exists
            kwargs.pop('low_cpu_mem_usage', None)
            kwargs['low_cpu_mem_usage'] = False
        
        # Load model
        load_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': config.trust_remote_code,
            'low_cpu_mem_usage': kwargs.get('low_cpu_mem_usage', config.low_cpu_mem_usage),
            'use_safetensors': True,  # Prefer safetensors format
            **{k: v for k, v in kwargs.items() if k not in ['low_cpu_mem_usage', 'device_map', 'use_safetensors']}
        }
        
        # Add quantization config if available
        if quantization_config is not None:
            load_kwargs['quantization_config'] = quantization_config
        
        # Only add device_map if it's not None
        if device_map is not None:
            load_kwargs['device_map'] = device_map
        
        # Add token if available
        if hf_token:
            load_kwargs['token'] = hf_token
        
        # Load model with error handling for authentication and network issues
        max_retries = 3
        retry_delay = 5  # seconds
        
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
                # Success - break out of retry loop
                break
                
            except Exception as e:
                error_str = str(e).lower()
                error_msg = str(e)
                
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
        if self.test_mode:
            return "cpu"  # Always use CPU in test mode
        
        if self.system_info.gpu_count == 0:
            return "cpu"
        
        # For production, use GPU if available
        if config.size in [ModelSize.TINY, ModelSize.SMALL]:
            return "auto"  # Let transformers decide
        else:
            return "auto"  # For large models, use auto mapping
    
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
