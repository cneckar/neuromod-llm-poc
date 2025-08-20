"""
Model Manager for Cloud Run Compatible Models
Handles model loading, caching, and resource management
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import logging
from typing import Dict, Any, Optional, List
import gc
import time

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model loading and caching for Cloud Run"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_type = None
        self.device = "cpu"  # Cloud Run doesn't support GPU
        
        # Cloud Run compatible models
        self.compatible_models = {
            # DialoGPT series (conversational)
            "microsoft/DialoGPT-small": {
                "type": "causal",
                "size_mb": 500,
                "max_length": 1024,
                "description": "Small conversational model, fast inference"
            },
            "microsoft/DialoGPT-medium": {
                "type": "causal", 
                "size_mb": 1500,
                "max_length": 1024,
                "description": "Medium conversational model, good balance"
            },
            "microsoft/DialoGPT-large": {
                "type": "causal",
                "size_mb": 3000,
                "max_length": 1024,
                "description": "Large conversational model, higher quality"
            },
            
            # GPT-2 series
            "gpt2": {
                "type": "causal",
                "size_mb": 500,
                "max_length": 1024,
                "description": "Original GPT-2, good for text generation"
            },
            "gpt2-medium": {
                "type": "causal",
                "size_mb": 1500,
                "max_length": 1024,
                "description": "Medium GPT-2, better quality"
            },
            
            # Distil variants (faster)
            "distilgpt2": {
                "type": "causal",
                "size_mb": 350,
                "max_length": 1024,
                "description": "Distilled GPT-2, faster inference"
            },
            
            # T5 series (text-to-text)
            "t5-small": {
                "type": "seq2seq",
                "size_mb": 240,
                "max_length": 512,
                "description": "Small T5, good for summarization"
            },
            "t5-base": {
                "type": "seq2seq",
                "size_mb": 850,
                "max_length": 512,
                "description": "Base T5, better quality"
            },
            
            # BART series
            "facebook/bart-base": {
                "type": "seq2seq",
                "size_mb": 500,
                "max_length": 1024,
                "description": "Base BART, good for text generation"
            }
        }
    
    def list_compatible_models(self) -> List[Dict[str, Any]]:
        """List all Cloud Run compatible models"""
        return [
            {
                "name": name,
                "type": info["type"],
                "size_mb": info["size_mb"],
                "max_length": info["max_length"],
                "description": info["description"]
            }
            for name, info in self.compatible_models.items()
        ]
    
    def is_model_compatible(self, model_name: str) -> bool:
        """Check if a model is compatible with Cloud Run"""
        return model_name in self.compatible_models
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.compatible_models:
            info = self.compatible_models[model_name].copy()
            info["name"] = model_name
            return info
        return None
    
    def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        try:
            # Check if model is compatible
            if not self.is_model_compatible(model_name):
                logger.error(f"Model {model_name} is not compatible with Cloud Run")
                return False
            
            # Get model info
            model_info = self.get_model_info(model_name)
            if not model_info:
                return False
            
            logger.info(f"Loading model: {model_name}")
            start_time = time.time()
            
            # Clear existing model if any
            self.unload_model()
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if model_info["type"] == "causal":
                logger.info("Loading causal model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            elif model_info["type"] == "seq2seq":
                logger.info("Loading seq2seq model...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Store model info
            self.model_name = model_name
            self.model_type = model_info["type"]
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.unload_model()
            return False
    
    def unload_model(self):
        """Unload current model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.model_name = None
        self.model_type = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Model unloaded and memory freed")
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded")
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True)
            
            # Generate
            with torch.no_grad():
                if self.model_type == "causal":
                    outputs = self.model.generate(
                        inputs,
                        max_length=inputs.shape[1] + max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                elif self.model_type == "seq2seq":
                    outputs = self.model.generate(
                        inputs,
                        max_length=max_tokens,
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
        return {
            "model_loaded": self.model is not None,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "compatible_models": len(self.compatible_models)
        }
    
    def estimate_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of current model"""
        if self.model is None:
            return {"model_memory_mb": 0, "total_memory_mb": 0}
        
        # Estimate model memory (rough calculation)
        model_params = sum(p.numel() for p in self.model.parameters())
        model_memory_mb = (model_params * 4) / (1024 * 1024)  # 4 bytes per parameter
        
        # Estimate total memory (model + overhead)
        total_memory_mb = model_memory_mb * 1.5  # Add 50% overhead
        
        return {
            "model_memory_mb": round(model_memory_mb, 2),
            "total_memory_mb": round(total_memory_mb, 2),
            "parameters": model_params
        }

# Global model manager instance
model_manager = ModelManager()
