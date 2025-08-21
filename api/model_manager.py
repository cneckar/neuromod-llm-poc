"""
Enhanced Model Manager for Local Models and Vertex AI Endpoints
Handles model loading, caching, and resource management with support for both local and remote models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import logging
from typing import Dict, Any, Optional, List, Union
import gc
import time
import requests
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> str:
        """Generate text with optional neuromodulation effects"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current model status"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model interface is available"""
        pass

class LocalModelInterface(BaseModelInterface):
    """Interface for local models using transformers"""
    
    def __init__(self, model_name: str, model_type: str = "causal"):
        self.model_name = model_name
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the local model"""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.model_type == "causal":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            elif self.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="cpu"
                )
            
            self.model.eval()
            logger.info(f"Local model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> str:
        """Generate text using local model (neuromodulation not supported for local models)"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model not loaded")
        
        # Note: Local models don't support neuromodulation effects
        if any([pack_name, custom_pack, individual_effects, multiple_packs]):
            logger.warning("Neuromodulation effects not supported for local models")
        
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
            logger.error(f"Local text generation failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get local model status"""
        return {
            "type": "local",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_loaded": self.model is not None,
            "device": self.device
        }
    
    def is_available(self) -> bool:
        """Check if local model is available"""
        return self.model is not None and self.tokenizer is not None
    
    def unload(self):
        """Unload local model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Local model unloaded and memory freed")

class VertexAIInterface(BaseModelInterface):
    """Interface for Vertex AI endpoints with neuromodulation support"""
    
    def __init__(self, endpoint_url: str, project_id: str, location: str = "us-central1"):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.project_id = project_id
        self.location = location
        self.auth_token = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Cloud"""
        try:
            import subprocess
            result = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True,
                text=True,
                check=True
            )
            self.auth_token = result.stdout.strip()
            logger.info("Successfully authenticated with Google Cloud")
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Cloud: {e}")
            raise
    
    def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to Vertex AI endpoint"""
        if not self.auth_token:
            self._authenticate()
        
        url = f"{self.endpoint_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to Vertex AI endpoint failed: {e}")
            raise
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> str:
        """Generate text using Vertex AI endpoint with neuromodulation support"""
        try:
            # Prepare request data
            request_data = {
                "instances": [{
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }]
            }
            
            # Add neuromodulation parameters
            if pack_name:
                request_data["instances"][0]["pack_name"] = pack_name
            elif custom_pack:
                request_data["instances"][0]["custom_pack"] = custom_pack
            elif individual_effects:
                request_data["instances"][0]["individual_effects"] = individual_effects
            elif multiple_packs:
                request_data["instances"][0]["multiple_packs"] = multiple_packs
            
            # Make request to Vertex AI endpoint
            response = self._make_request("/predict", request_data)
            
            # Extract generated text
            if "predictions" in response and len(response["predictions"]) > 0:
                prediction = response["predictions"][0]
                if "generated_text" in prediction:
                    return prediction["generated_text"]
                elif "error" in prediction:
                    raise RuntimeError(f"Vertex AI error: {prediction['error']}")
                else:
                    raise RuntimeError("Unexpected response format from Vertex AI")
            else:
                raise RuntimeError("No predictions in Vertex AI response")
                
        except Exception as e:
            logger.error(f"Vertex AI text generation failed: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """Get Vertex AI endpoint status"""
        try:
            response = self._make_request("/health", {})
            return {
                "type": "vertex_ai",
                "endpoint_url": self.endpoint_url,
                "project_id": self.project_id,
                "location": self.location,
                "status": response.get("status", "unknown"),
                "model_loaded": response.get("model_loaded", False),
                "model_name": response.get("model_name", "unknown")
            }
        except Exception as e:
            return {
                "type": "vertex_ai",
                "endpoint_url": self.endpoint_url,
                "project_id": self.project_id,
                "location": self.location,
                "status": "error",
                "error": str(e)
            }
    
    def is_available(self) -> bool:
        """Check if Vertex AI endpoint is available"""
        try:
            status = self.get_status()
            return status.get("status") == "healthy" and status.get("model_loaded", False)
        except Exception:
            return False
    
    def get_available_packs(self) -> List[str]:
        """Get available neuromodulation packs from Vertex AI endpoint"""
        try:
            response = self._make_request("/available_packs", {})
            if "available_packs" in response:
                return response["available_packs"]
            return []
        except Exception as e:
            logger.error(f"Failed to get available packs: {e}")
            return []
    
    def get_available_effects(self) -> List[str]:
        """Get available individual effects from Vertex AI endpoint"""
        try:
            response = self._make_request("/available_effects", {})
            if "available_effects" in response:
                return response["available_effects"]
            return []
        except Exception as e:
            logger.error(f"Failed to get available effects: {e}")
            return []

class EnhancedModelManager:
    """Enhanced model manager supporting both local and Vertex AI models"""
    
    def __init__(self):
        self.current_interface: Optional[BaseModelInterface] = None
        self.interface_type: Optional[str] = None
        
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
    
    def load_local_model(self, model_name: str) -> bool:
        """Load a local model"""
        try:
            if not self.is_model_compatible(model_name):
                logger.error(f"Model {model_name} is not compatible with Cloud Run")
                return False
            
            model_info = self.get_model_info(model_name)
            if not model_info:
                return False
            
            # Unload current interface if any
            self.unload_current_interface()
            
            # Create and load local interface
            self.current_interface = LocalModelInterface(model_name, model_info["type"])
            self.interface_type = "local"
            
            logger.info(f"Local model {model_name} loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            self.unload_current_interface()
            return False
    
    def connect_vertex_ai(self, endpoint_url: str, project_id: str, location: str = "us-central1") -> bool:
        """Connect to a Vertex AI endpoint"""
        try:
            # Unload current interface if any
            self.unload_current_interface()
            
            # Create and test Vertex AI interface
            self.current_interface = VertexAIInterface(endpoint_url, project_id, location)
            
            # Test connection
            if self.current_interface.is_available():
                self.interface_type = "vertex_ai"
                logger.info(f"Connected to Vertex AI endpoint: {endpoint_url}")
                return True
            else:
                logger.error("Vertex AI endpoint is not available")
                self.unload_current_interface()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Vertex AI endpoint: {e}")
            self.unload_current_interface()
            return False
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> str:
        """Generate text using current interface with neuromodulation support"""
        if self.current_interface is None:
            raise RuntimeError("No model interface loaded")
        
        return self.current_interface.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pack_name=pack_name,
            custom_pack=custom_pack,
            individual_effects=individual_effects,
            multiple_packs=multiple_packs
        )
    
    def get_available_packs(self) -> List[str]:
        """Get available neuromodulation packs"""
        if self.interface_type == "vertex_ai" and self.current_interface:
            return self.current_interface.get_available_packs()
        elif self.interface_type == "local":
            # Local models don't support neuromodulation
            return []
        else:
            return []
    
    def get_available_effects(self) -> List[str]:
        """Get available individual effects"""
        if self.interface_type == "vertex_ai" and self.current_interface:
            return self.current_interface.get_available_effects()
        elif self.interface_type == "local":
            # Local models don't support neuromodulation
            return []
        else:
            return []
    
    def unload_current_interface(self):
        """Unload current interface"""
        if self.current_interface:
            if self.interface_type == "local":
                self.current_interface.unload()
            self.current_interface = None
            self.interface_type = None
            logger.info("Current interface unloaded")
    
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        status = {
            "interface_type": self.interface_type,
            "interface_available": self.current_interface is not None
        }
        
        if self.current_interface:
            status.update(self.current_interface.get_status())
        
        return status
    
    def is_available(self) -> bool:
        """Check if any interface is available"""
        return self.current_interface is not None and self.current_interface.is_available()

# Global enhanced model manager instance
enhanced_model_manager = EnhancedModelManager()

# Legacy compatibility - keep the old model_manager for backward compatibility
class ModelManager:
    """Legacy model manager for backward compatibility"""
    
    def __init__(self):
        self._enhanced = enhanced_model_manager
    
    def load_model(self, model_name: str) -> bool:
        """Load a local model (legacy method)"""
        return self._enhanced.load_local_model(model_name)
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0) -> str:
        """Generate text (legacy method)"""
        return self._enhanced.generate_text(prompt, max_tokens, temperature, top_p)
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get model status (legacy method)"""
        return self._enhanced.get_status()
    
    def unload_model(self):
        """Unload model (legacy method)"""
        self._enhanced.unload_current_interface()

# Create legacy instance for backward compatibility
model_manager = ModelManager()
