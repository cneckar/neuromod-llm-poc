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
                "meta-llama/Llama-3.1-8B": ModelConfig(
                    name="meta-llama/Llama-3.1-8B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "meta-llama/Llama-3.1-70B": ModelConfig(
                    name="meta-llama/Llama-3.1-70B",
                    size=ModelSize.LARGE,
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
                "Qwen/Qwen-2.5-Omni-7B": ModelConfig(
                    name="Qwen/Qwen-2.5-Omni-7B",
                    size=ModelSize.SMALL,
                    backend=BackendType.HUGGINGFACE,
                    quantization="8bit",
                    max_length=2048,
                    torch_dtype=torch.float16
                ),
                "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelConfig(
                    name="mistralai/Mixtral-8x22B-Instruct-v0.1",
                    size=ModelSize.XLARGE,
                    backend=BackendType.HUGGINGFACE,
                    quantization="4bit",
                    max_length=2048,
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
            if self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 40:
                return "meta-llama/Llama-3.1-70B"
            elif self.system_info.gpu_memory_gb and self.system_info.gpu_memory_gb > 16:
                return "meta-llama/Llama-3.1-8B"
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
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Determine device and dtype
        device_map = self._get_device_map(config)
        torch_dtype = config.torch_dtype or torch.float32
        
        # Check if MPS is available to avoid accelerate issues
        if not hasattr(torch, 'mps') or not torch.backends.mps.is_available():
            # Force CPU loading to avoid accelerate MPS issues
            device_map = None  # Don't use device_map at all
            kwargs.pop('device_map', None)  # Remove device_map from kwargs
            # Disable accelerate to avoid MPS issues
            kwargs['use_safetensors'] = False
            # Override low_cpu_mem_usage if it exists
            kwargs.pop('low_cpu_mem_usage', None)
            kwargs['low_cpu_mem_usage'] = False
        
        # Load model
        load_kwargs = {
            'torch_dtype': torch_dtype,
            'trust_remote_code': config.trust_remote_code,
            'low_cpu_mem_usage': kwargs.get('low_cpu_mem_usage', config.low_cpu_mem_usage),
            **{k: v for k, v in kwargs.items() if k != 'low_cpu_mem_usage'}
        }
        
        # Only add device_map if it's not None
        if device_map is not None:
            load_kwargs['device_map'] = device_map
        
        model = AutoModelForCausalLM.from_pretrained(
            config.name,
            **load_kwargs
        )
        
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
        "meta-llama/Llama-3.1-70B": "model.layers.{}.self_attn",
        "Qwen/Qwen-2.5-7B": "model.layers.{}.self_attn",
        "Qwen/Qwen-2.5-Omni-7B": "model.layers.{}.self_attn",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "model.layers.{}.self_attn"
    }
    
    # Find matching pattern
    for pattern, path in hook_paths.items():
        if pattern in model_name:
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
