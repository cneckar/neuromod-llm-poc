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
import os

# Disable MPS completely for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neuromodulation components
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from neuromod import NeuromodTool
    from neuromod.effects import EffectRegistry
    from neuromod.pack_system import Pack, EffectConfig
    from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker
    NEUROMOD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neuromodulation system not available: {e}")
    NEUROMOD_AVAILABLE = False

class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> Dict[str, Any]:
        """Generate text with optional neuromodulation effects and emotion tracking"""
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
    """Interface for local models using the centralized model support system"""
    
    def __init__(self, model_name: str, model_type: str = "causal", test_mode: bool = True):
        self.model_name = model_name
        self.model_type = model_type
        self.test_mode = test_mode
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.neuromod_tool = None
        self.model_manager = None
        
        # Initialize emotion tracking
        self.emotion_tracker = SimpleEmotionTracker()
        self.session_id = f"api_session_{int(time.time())}"
        
        self._load_model()
        self._setup_neuromodulation()
    
    def _load_model(self):
        """Load the local model using centralized model support"""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            
            # Import model support system
            from neuromod.model_support import create_model_support
            
            # Create model manager
            self.model_manager = create_model_support(test_mode=self.test_mode)
            
            # Load model using centralized system
            self.model, self.tokenizer, model_info = self.model_manager.load_model(
                self.model_name
            )
            
            # Set device info
            self.device = model_info.get('device_map', 'cpu')
            
            logger.info(f"Local model {self.model_name} loaded successfully")
            logger.info(f"Model info: {model_info}")
            
        except Exception as e:
            logger.error(f"Failed to load local model {self.model_name}: {e}")
            raise
    
    def _setup_neuromodulation(self):
        """Initialize neuromodulation system for local model"""
        try:
            if NEUROMOD_AVAILABLE:
                # Initialize neuromodulation tool with the loaded model
                from neuromod.pack_system import PackRegistry
                registry = PackRegistry("packs/config.json")
                self.neuromod_tool = NeuromodTool(registry, self.model, self.tokenizer)
                logger.info("Neuromodulation tool initialized for local model")
            else:
                logger.info("Neuromodulation system not available - basic text generation only")
                self.neuromod_tool = None
        except Exception as e:
            logger.info(f"Neuromodulation not available: {e}")
            self.neuromod_tool = None
    
    def generate_text(self, prompt: str, max_tokens: int = 100, 
                     temperature: float = 1.0, top_p: float = 1.0,
                     pack_name: str = None, custom_pack: Dict = None,
                     individual_effects: List[Dict] = None,
                     multiple_packs: List[str] = None) -> str:
        """Generate text using local model with optional neuromodulation support"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model not loaded")
        
        try:
            # Apply neuromodulation effects if available
            neuromod_applied = False
            if self.neuromod_tool and any([pack_name, custom_pack, individual_effects, multiple_packs]):
                try:
                    neuromod_applied = self._apply_neuromodulation(
                        pack_name=pack_name,
                        custom_pack=custom_pack,
                        individual_effects=individual_effects,
                        multiple_packs=multiple_packs
                    )
                    
                    if neuromod_applied:
                        logger.info("Neuromodulation effects applied to local model")
                except Exception as e:
                    logger.warning(f"Neuromodulation failed, continuing with basic generation: {e}")
                    neuromod_applied = False
            
            # Tokenize input with simple, safe approach (same as working neuromod tests)
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the same device as the model
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            else:
                inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Get neuromodulation effects if available
            logits_processors = []
            gen_kwargs = {}
            
            if self.neuromod_tool:
                # Update token position for phase-based effects
                self.neuromod_tool.update_token_position(0)  # Reset for each new prompt
                logits_processors = self.neuromod_tool.get_logits_processors()
                gen_kwargs = self.neuromod_tool.get_generation_kwargs()
            
            # Generate text with safe settings to avoid bus errors
            with torch.no_grad():
                if self.model_type == "causal":
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                        early_stopping=False,
                        logits_processor=logits_processors,
                        **gen_kwargs
                    )
                elif self.model_type == "seq2seq":
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=False,
                        repetition_penalty=1.1,
                        no_repeat_ngram_size=2,
                        early_stopping=False,
                        logits_processor=logits_processors,
                        **gen_kwargs
                    )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            # Track emotions for the generated response
            emotion_data = self._track_emotions(generated_text, prompt)
            
            # Clear neuromodulation effects after generation (if any were applied)
            if neuromod_applied and self.neuromod_tool:
                try:
                    self.neuromod_tool.clear()
                    logger.info("Neuromodulation effects cleared from local model")
                except Exception as e:
                    logger.warning(f"Failed to clear neuromodulation effects: {e}")
            
            # Return both text and emotion data
            return {
                "text": generated_text,
                "emotions": emotion_data
            }
            
        except Exception as e:
            logger.error(f"Local text generation failed: {e}")
            # Try to clear effects on error
            if self.neuromod_tool:
                try:
                    self.neuromod_tool.clear()
                except:
                    pass
            raise
    
    def _apply_neuromodulation(self, pack_name: str = None, custom_pack: Dict = None,
                               individual_effects: List[Dict] = None,
                               multiple_packs: List[str] = None) -> bool:
        """Apply neuromodulation effects to the local model"""
        if not self.neuromod_tool:
            logger.warning("Neuromodulation tool not available")
            return False
        
        try:
            # Clear any existing effects first
            self.neuromod_tool.clear()
            
            # Method 1: Single predefined pack
            if pack_name:
                self.neuromod_tool.load_pack(pack_name)
                logger.info(f"Applied predefined pack: {pack_name}")
            
            # Method 2: Custom pack definition
            elif custom_pack:
                # Create Pack object from custom definition
                effects = [EffectConfig.from_dict(effect) for effect in custom_pack.get('effects', [])]
                pack = Pack(
                    name=custom_pack.get('name', 'custom'),
                    description=custom_pack.get('description', 'Custom neuromodulation pack'),
                    effects=effects
                )
                # Apply using the pack manager directly
                self.neuromod_tool.pack_manager.apply_pack(pack, self.model)
                logger.info(f"Applied custom pack: {pack.name}")
            
            # Method 3: Individual effects
            elif individual_effects:
                for effect_data in individual_effects:
                    effect_name = effect_data.get('effect')
                    weight = effect_data.get('weight', 0.5)
                    direction = effect_data.get('direction', 'up')
                    parameters = effect_data.get('parameters', {})
                    
                    # Create effect config
                    effect_config = EffectConfig(
                        effect=effect_name,
                        weight=weight,
                        direction=direction,
                        parameters=parameters
                    )
                    
                    # Create a single-effect pack and apply it
                    single_pack = Pack(
                        name=f"single_{effect_name}",
                        description=f"Single effect: {effect_name}",
                        effects=[effect_config]
                    )
                    self.neuromod_tool.pack_manager.apply_pack(single_pack, self.model)
                    logger.info(f"Applied individual effect: {effect_name}")
            
            # Method 4: Multiple packs (combine effects)
            elif multiple_packs:
                all_effects = []
                for pack_name in multiple_packs:
                    try:
                        # Load pack to get effects
                        self.neuromod_tool.load_pack(pack_name)
                        # Get the active effects and add to our list
                        pack_info = self.neuromod_tool.get_effect_info()
                        logger.info(f"Loaded pack: {pack_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load pack {pack_name}: {e}")
                
                logger.info(f"Applied multiple packs: {multiple_packs}")
            
            # Apply effects to the model through the pack manager
            if any([pack_name, custom_pack, individual_effects, multiple_packs]):
                # The effects are applied through the pack manager and will be used during generation
                logger.info("Applied neuromodulation effects to local model")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to apply neuromodulation to local model: {e}")
            return False
    
    def _track_emotions(self, response: str, context: str = "") -> Dict[str, Any]:
        """Track emotions for the generated response"""
        try:
            # Track emotion changes
            latest_state = self.emotion_tracker.assess_emotion_change(response, self.session_id, context)
            
            if latest_state:
                # Extract emotion data
                emotion_changes = {}
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']:
                    emotion_value = getattr(latest_state, emotion)
                    emotion_changes[emotion] = emotion_value
                
                return {
                    "current_state": emotion_changes,
                    "valence": latest_state.valence,
                    "confidence": latest_state.confidence,
                    "timestamp": latest_state.timestamp
                }
            else:
                return {
                    "current_state": {},
                    "valence": "neutral",
                    "confidence": 0.0,
                    "timestamp": None
                }
                
        except Exception as e:
            logger.warning(f"Emotion tracking error: {e}")
            return {
                "current_state": {},
                "valence": "neutral",
                "confidence": 0.0,
                "timestamp": None,
                "error": str(e)
            }
    
    def get_available_packs(self) -> List[str]:
        """Get available neuromodulation packs for local model"""
        if self.neuromod_tool:
            try:
                return self.neuromod_tool.registry.list_packs()
            except Exception as e:
                logger.error(f"Failed to get available packs: {e}")
                return []
        return []
    
    def get_available_effects(self) -> List[str]:
        """Get available individual effects for local model"""
        if self.neuromod_tool:
            try:
                return self.neuromod_tool.pack_manager.effect_registry.list_effects()
            except Exception as e:
                logger.error(f"Failed to get available effects: {e}")
                return []
        return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get local model status with neuromodulation info"""
        status = {
            "type": "local",
            "model_name": self.model_name,
            "model_type": self.model_type,
            "model_loaded": self.model is not None,
            "device": self.device,
            "neuromodulation_available": self.neuromod_tool is not None
        }
        
        if self.neuromod_tool:
            status["available_packs"] = len(self.get_available_packs())
            status["available_effects"] = len(self.get_available_effects())
        
        return status
    
    def is_available(self) -> bool:
        """Check if local model is available"""
        return self.model is not None and self.tokenizer is not None
    
    def cleanup(self):
        """Clean up model resources"""
        if self.model_manager:
            self.model_manager.cleanup()
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        logger.info(f"Cleaned up model interface for {self.model_name}")
    
    def unload(self):
        """Unload local model and free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear neuromodulation tool
        if self.neuromod_tool:
            self.neuromod_tool.clear()
            self.neuromod_tool = None
        
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
                    generated_text = prediction["generated_text"]
                    
                    # For Vertex AI, we'll return basic emotion structure
                    # (emotion tracking would need to be implemented on the endpoint side)
                    emotion_data = {
                        "current_state": {},
                        "valence": "neutral",
                        "confidence": 0.0,
                        "timestamp": None,
                        "note": "Emotion tracking not available for Vertex AI endpoints"
                    }
                    
                    return {
                        "text": generated_text,
                        "emotions": emotion_data
                    }
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
            # Llama series (high quality)
            "meta-llama/Llama-3.1-1B-Instruct": {
                "type": "causal",
                "size_mb": 2000,
                "max_length": 8192,
                "description": "Llama 3.1 1B - Fast and efficient, great for testing"
            },
            "meta-llama/Llama-3.1-3B-Instruct": {
                "type": "causal",
                "size_mb": 6000,
                "max_length": 8192,
                "description": "Llama 3.1 3B - Good balance of speed and quality"
            },
            "meta-llama/Llama-3.1-8B-Instruct": {
                "type": "causal",
                "size_mb": 16000,
                "max_length": 8192,
                "description": "Llama 3.1 8B - Excellent quality, needs 10-12GB VRAM"
            },
            "meta-llama/Llama-3.1-70B-Instruct": {
                "type": "causal",
                "size_mb": 140000,
                "max_length": 8192,
                "description": "Llama 3.1 70B - Best quality, needs quantization for RTX 4070"
            },
            "meta-llama/Llama-2-7B-Chat-HF": {
                "type": "causal",
                "size_mb": 14000,
                "max_length": 4096,
                "description": "Llama 2 7B - Good quality, fits well on RTX 4070"
            },
            "meta-llama/Llama-2-13B-Chat-HF": {
                "type": "causal",
                "size_mb": 26000,
                "max_length": 4096,
                "description": "Llama 2 13B - Better quality, fits on RTX 4070"
            },
            
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
            },

            # OpenAI GPT-OSS reference models (community weights)
            "openai/gpt-oss-20b": {
                "type": "causal",
                "size_mb": 40000,
                "max_length": 8192,
                "description": "OpenAI GPT-OSS 20B - high-fidelity open checkpoint (4-bit recommended)"
            },
            "openai/gpt-oss-120b": {
                "type": "causal",
                "size_mb": 240000,
                "max_length": 8192,
                "description": "OpenAI GPT-OSS 120B - flagship open release, requires multi-GPU or >80GB VRAM"
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
        elif self.interface_type == "local" and self.current_interface:
            # Local models now support neuromodulation
            return self.current_interface.get_available_packs()
        else:
            return []
    
    def get_available_effects(self) -> List[str]:
        """Get available individual effects"""
        if self.interface_type == "vertex_ai" and self.current_interface:
            return self.current_interface.get_available_effects()
        elif self.interface_type == "local" and self.current_interface:
            # Local models now support neuromodulation
            return self.current_interface.get_available_effects()
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
    
    @property
    def model(self):
        """Legacy property for backward compatibility"""
        status = self.get_model_status()
        return status.get("interface_available", False)

# Create legacy instance for backward compatibility
model_manager = ModelManager()
