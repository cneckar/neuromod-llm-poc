"""
GPU Model Manager for Large Language Models
Handles Llama 3, Qwen, Mixtral and other GPU-required models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, Any, Optional, List
import gc
import time
import os

logger = logging.getLogger(__name__)

class GPUModelManager:
    """Manages GPU model loading and inference for large models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        
        # Check GPU availability
        if torch.cuda.is_available():
            self.device = "cuda"
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)} ({self.gpu_memory:.1f}GB)")
        else:
            self.device = "cpu"
            logger.warning("No GPU available, falling back to CPU")
        
        # GPU-compatible large models
        self.gpu_models = {
            # Llama 3 series
            "meta-llama/Meta-Llama-3.1-8B": {
                "type": "causal",
                "size_gb": 8,
                "quantized_size_gb": 4,  # 4-bit quantization
                "max_length": 8192,
                "description": "Llama 3.1 8B - Good balance of quality and speed"
            },
            "meta-llama/Meta-Llama-3.1-70B": {
                "type": "causal", 
                "size_gb": 70,
                "quantized_size_gb": 35,  # 4-bit quantization
                "max_length": 8192,
                "description": "Llama 3.1 70B - High quality, requires 24GB+ VRAM"
            },
            
            # Qwen series
            "Qwen/Qwen2.5-7B": {
                "type": "causal",
                "size_gb": 7,
                "quantized_size_gb": 3.5,
                "max_length": 32768,
                "description": "Qwen 2.5 7B - Good performance, reasonable size"
            },
            "Qwen/Qwen2.5-32B": {
                "type": "causal",
                "size_gb": 32,
                "quantized_size_gb": 16,
                "max_length": 32768,
                "description": "Qwen 2.5 32B - High quality, requires 16GB+ VRAM"
            },
            
            # Mixtral series (MoE models)
            "mistralai/Mixtral-8x7B-v0.1": {
                "type": "causal",
                "size_gb": 8,
                "quantized_size_gb": 4,
                "max_length": 32768,
                "description": "Mixtral 8x7B - MoE model, good quality"
            },
            "mistralai/Mixtral-8x22B-v0.1": {
                "type": "causal",
                "size_gb": 22,
                "quantized_size_gb": 11,
                "max_length": 65536,
                "description": "Mixtral 8x22B - Large MoE model, high quality"
            },
            
            # Other high-quality models
            "microsoft/Phi-3.5-14B": {
                "type": "causal",
                "size_gb": 14,
                "quantized_size_gb": 7,
                "max_length": 8192,
                "description": "Phi 3.5 14B - Microsoft's latest model"
            },
            "google/gemma-2-9B": {
                "type": "causal",
                "size_gb": 9,
                "quantized_size_gb": 4.5,
                "max_length": 8192,
                "description": "Gemma 2 9B - Google's open model"
            }
        }
    
    def list_gpu_models(self) -> List[Dict[str, Any]]:
        """List all GPU-compatible models"""
        return [
            {
                "name": name,
                "type": info["type"],
                "size_gb": info["size_gb"],
                "quantized_size_gb": info["quantized_size_gb"],
                "max_length": info["max_length"],
                "description": info["description"],
                "compatible": self._is_model_compatible(info)
            }
            for name, info in self.gpu_models.items()
        ]
    
    def _is_model_compatible(self, model_info: Dict[str, Any]) -> bool:
        """Check if model can fit in available GPU memory"""
        if self.device == "cpu":
            return False  # Large models need GPU
        
        required_memory = model_info["quantized_size_gb"]
        return required_memory <= self.gpu_memory
    
    def is_model_compatible(self, model_name: str) -> bool:
        """Check if a specific model is compatible"""
        if model_name not in self.gpu_models:
            return False
        return self._is_model_compatible(self.gpu_models[model_name])
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.gpu_models:
            info = self.gpu_models[model_name].copy()
            info["name"] = model_name
            info["compatible"] = self._is_model_compatible(info)
            return info
        return None
    
    def load_model(self, model_name: str, quantization: str = "4bit") -> bool:
        """Load a model with quantization"""
        try:
            # Check if model is compatible
            if not self.is_model_compatible(model_name):
                logger.error(f"Model {model_name} is not compatible with current GPU")
                return False
            
            # Get model info
            model_info = self.get_model_info(model_name)
            if not model_info:
                return False
            
            logger.info(f"Loading model: {model_name} with {quantization} quantization")
            start_time = time.time()
            
            # Clear existing model if any
            self.unload_model()
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization
            logger.info(f"Loading model with {quantization} quantization...")
            
            if quantization == "4bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            elif quantization == "8bit":
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            else:  # No quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
            
            # Store model info
            self.model_name = model_name
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.unload_model()
            return False
    
    def unload_model(self):
        """Unload current model and free GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.model_name = None
        
        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Model unloaded and GPU memory freed")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            
            # Move to GPU if available
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            raise
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9  # GB
        
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "device": self.device,
            "gpu_memory_total_gb": self.gpu_memory if self.device == "cuda" else 0,
            "gpu_memory_used_gb": round(gpu_memory_used, 2),
            "compatible_models": len([m for m in self.gpu_models.values() if self._is_model_compatible(m)])
        }
    
    def estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of current model"""
        if self.model is None:
            return {"model_memory_gb": 0, "gpu_memory_used_gb": 0}
        
        # Get actual GPU memory usage
        gpu_memory_used = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / 1e9  # GB
        
        # Estimate model memory from model info
        model_memory_gb = 0
        if self.model_name in self.gpu_models:
            model_memory_gb = self.gpu_models[self.model_name]["quantized_size_gb"]
        
        return {
            "model_memory_gb": model_memory_gb,
            "gpu_memory_used_gb": round(gpu_memory_used, 2),
            "gpu_memory_available_gb": round(self.gpu_memory - gpu_memory_used, 2)
        }

# Global GPU model manager instance
gpu_model_manager = GPUModelManager()
