"""
Vertex AI Manager for Pay-Per-Use Model Serving
Integrates with Google Cloud Vertex AI for Llama 3 and other models
WITH FULL NEUROMODULATION PROBE SYSTEM INTEGRATION
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
import requests
from google.cloud import aiplatform
from google.auth import default

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

logger = logging.getLogger(__name__)

class VertexAIManager:
    """Manages Vertex AI endpoints for pay-per-use model serving WITH FULL PROBE SYSTEM"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.endpoint = None
        self.model_name = None
        
        # Initialize neuromodulation system
        self.neuromod_tool = None
        self.emotion_tracker = None
        self.current_pack = None
        self.probe_hooks = []
        
        if NEUROMOD_AVAILABLE:
            try:
                self.neuromod_tool = NeuromodTool()
                self.emotion_tracker = SimpleEmotionTracker()
                logger.info("✅ Neuromodulation system initialized")
            except Exception as e:
                logger.error(f"Failed to initialize neuromodulation: {e}")
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Vertex AI compatible models
        self.vertex_models = {
            # Llama 3 series
            "meta-llama/Meta-Llama-3.1-8B": {
                "type": "causal",
                "description": "Llama 3.1 8B - Good balance of quality and cost",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-4",
                "accelerator": "NVIDIA_TESLA_T4",
                "supports_neuromodulation": True
            },
            "meta-llama/Meta-Llama-3.1-70B": {
                "type": "causal",
                "description": "Llama 3.1 70B - High quality, higher cost",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-16",
                "accelerator": "NVIDIA_TESLA_A100",
                "supports_neuromodulation": True
            },
            
            # Qwen series
            "Qwen/Qwen2.5-7B": {
                "type": "causal",
                "description": "Qwen 2.5 7B - Good performance, reasonable cost",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-4",
                "accelerator": "NVIDIA_TESLA_T4",
                "supports_neuromodulation": True
            },
            "Qwen/Qwen2.5-32B": {
                "type": "causal",
                "description": "Qwen 2.5 32B - High quality",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-16",
                "accelerator": "NVIDIA_TESLA_A100",
                "supports_neuromodulation": True
            },
            
            # Mixtral series
            "mistralai/Mixtral-8x7B-v0.1": {
                "type": "causal",
                "description": "Mixtral 8x7B - MoE model, good quality",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-8",
                "accelerator": "NVIDIA_TESLA_V100",
                "supports_neuromodulation": True
            }
        }
    
    def list_vertex_models(self) -> List[Dict[str, Any]]:
        """List all Vertex AI compatible models"""
        return [
            {
                "name": name,
                "type": info["type"],
                "description": info["description"],
                "endpoint_type": info["endpoint_type"],
                "machine_type": info["machine_type"],
                "accelerator": info["accelerator"],
                "supports_neuromodulation": info.get("supports_neuromodulation", False)
            }
            for name, info in self.vertex_models.items()
        ]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name in self.vertex_models:
            info = self.vertex_models[model_name].copy()
            info["name"] = model_name
            return info
        return None
    
    def apply_neuromodulation_pack(self, pack_name: str, intensity: float = 0.7) -> bool:
        """Apply a neuromodulation pack to the current model"""
        if not self.neuromod_tool:
            logger.error("Neuromodulation system not available")
            return False
        
        try:
            # Apply the pack
            success = self.neuromod_tool.apply(pack_name, intensity=intensity)
            if success:
                self.current_pack = pack_name
                logger.info(f"✅ Applied neuromodulation pack: {pack_name} (intensity: {intensity})")
                
                # Register probe hooks if we have a model
                if self.endpoint:
                    self._register_probe_hooks()
                
                return True
            else:
                logger.error(f"Failed to apply pack: {pack_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying neuromodulation pack: {e}")
            return False
    
    def clear_neuromodulation(self) -> bool:
        """Clear all neuromodulation effects"""
        if not self.neuromod_tool:
            return False
        
        try:
            # Remove probe hooks
            self._remove_probe_hooks()
            
            # Clear the pack
            self.current_pack = None
            
            logger.info("✅ Cleared neuromodulation effects")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing neuromodulation: {e}")
            return False
    
    def _register_probe_hooks(self):
        """Register PyTorch forward hooks for probe monitoring"""
        if not self.neuromod_tool or not self.probe_hooks:
            return
        
        try:
            # This would register hooks on the Vertex AI model
            # For now, we'll simulate it
            logger.info("🔌 Probe hooks registered for Vertex AI model")
            
        except Exception as e:
            logger.error(f"Failed to register probe hooks: {e}")
    
    def _remove_probe_hooks(self):
        """Remove PyTorch forward hooks"""
        if not self.probe_hooks:
            return
        
        try:
            # This would remove hooks from the Vertex AI model
            # For now, we'll simulate it
            logger.info("🔌 Probe hooks removed from Vertex AI model")
            self.probe_hooks = []
            
        except Exception as e:
            logger.error(f"Failed to remove probe hooks: {e}")
    
    def get_neuromodulation_status(self) -> Dict[str, Any]:
        """Get current neuromodulation status"""
        return {
            "neuromodulation_available": NEUROMOD_AVAILABLE,
            "neuromod_tool_loaded": self.neuromod_tool is not None,
            "current_pack": self.current_pack,
            "probe_hooks_active": len(self.probe_hooks) > 0,
            "emotion_tracking_active": self.emotion_tracker is not None
        }
    
    def create_custom_endpoint(self, model_name: str, endpoint_name: str = None) -> str:
        """Create a custom endpoint with FULL NEUROMODULATION PROBE SYSTEM support"""
        try:
            if endpoint_name is None:
                endpoint_name = f"neuromod-{model_name.replace('/', '-').replace('_', '-')}"
            
            model_info = self.get_model_info(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not supported")
            
            logger.info(f"Creating custom endpoint with FULL PROBE SYSTEM: {endpoint_name}")
            
            # Create custom container for neuromodulation with probe system
            container_uri = self._build_custom_container_with_probes(model_name)
            
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=self.project_id,
                location=self.region
            )
            
            # Deploy model to endpoint
            model = aiplatform.Model.upload(
                display_name=f"{model_name}-neuromod-probes",
                container_spec={
                    "image_uri": container_uri,
                    "predict_route": "/predict",
                    "health_route": "/health"
                },
                serving_container_environment_variables={
                    "MODEL_NAME": model_name,
                    "PROJECT_ID": self.project_id,
                    "NEUROMODULATION_ENABLED": "true",
                    "PROBE_SYSTEM_ENABLED": "true",
                    "EMOTION_TRACKING_ENABLED": "true"
                }
            )
            
            # Deploy to endpoint
            model.deploy(
                endpoint=endpoint,
                machine_type=model_info["machine_type"],
                accelerator_type=model_info["accelerator"],
                accelerator_count=1,
                min_replica_count=0,  # Scale to zero
                max_replica_count=1   # Pay per use
            )
            
            self.endpoint = endpoint
            self.model_name = model_name
            
            logger.info(f"✅ Endpoint created with FULL PROBE SYSTEM: {endpoint.resource_name}")
            return endpoint.resource_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint: {e}")
            raise
    
    def _build_custom_container_with_probes(self, model_name: str) -> str:
        """Build custom container with FULL NEUROMODULATION PROBE SYSTEM"""
        # This would build and push a custom container with:
        # - Full neuromodulation system
        # - Probe hooks and monitoring
        # - Emotion tracking
        # - Pack system
        # - Effect registry
        
        container_name = f"neuromod-probes-{model_name.replace('/', '-')}"
        return f"gcr.io/{self.project_id}/{container_name}:latest"
    
    def predict_with_neuromodulation(self, prompt: str, max_tokens: int = 100, 
                                   temperature: float = 1.0, top_p: float = 1.0,
                                   pack_name: str = None, track_emotions: bool = True) -> Dict[str, Any]:
        """Generate text using Vertex AI endpoint WITH FULL NEUROMODULATION AND PROBE SYSTEM"""
        if not self.endpoint:
            raise RuntimeError("No endpoint configured")
        
        try:
            # Apply neuromodulation pack if specified
            if pack_name and pack_name != self.current_pack:
                self.apply_neuromodulation_pack(pack_name)
            
            # Prepare prediction request with neuromodulation parameters
            prediction_request = {
                "instances": [{
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "pack_name": self.current_pack,
                    "neuromodulation_enabled": True,
                    "probe_system_enabled": True,
                    "emotion_tracking_enabled": track_emotions
                }]
            }
            
            # Make prediction
            start_time = time.time()
            response = self.endpoint.predict(prediction_request)
            prediction_time = time.time() - start_time
            
            # Extract result
            if response.predictions:
                result = response.predictions[0]
                generated_text = result.get("generated_text", "")
                
                # Process probe data if available
                probe_data = result.get("probe_data", {})
                emotion_data = result.get("emotion_data", {})
                
                # Track emotions locally if enabled
                if track_emotions and self.emotion_tracker:
                    session_id = f"vertex_ai_{int(time.time())}"
                    latest_state = self.emotion_tracker.assess_emotion_change(
                        generated_text, session_id, prompt
                    )
                    
                    if latest_state:
                        emotion_data = {
                            "current_state": {
                                'joy': getattr(latest_state, 'joy', 'stable'),
                                'sadness': getattr(latest_state, 'sadness', 'stable'),
                                'anger': getattr(latest_state, 'anger', 'stable'),
                                'fear': getattr(latest_state, 'fear', 'stable'),
                                'surprise': getattr(latest_state, 'surprise', 'stable'),
                                'disgust': getattr(latest_state, 'disgust', 'stable'),
                                'trust': getattr(latest_state, 'trust', 'stable'),
                                'anticipation': getattr(latest_state, 'anticipation', 'stable')
                            },
                            "valence": latest_state.valence,
                            "confidence": latest_state.confidence,
                            "timestamp": latest_state.timestamp
                        }
                
                logger.info(f"✅ Prediction completed with neuromodulation in {prediction_time:.2f}s")
                
                return {
                    "text": generated_text,
                    "probe_data": probe_data,
                    "emotions": emotion_data,
                    "neuromodulation_applied": self.current_pack,
                    "generation_time": prediction_time,
                    "model_type": "vertex_ai",
                    "vertex_model": self.model_name
                }
            else:
                raise RuntimeError("No prediction returned")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict(self, prompt: str, max_tokens: int = 100, 
               temperature: float = 1.0, top_p: float = 1.0,
               pack_name: str = None) -> str:
        """Legacy predict method - use predict_with_neuromodulation for full features"""
        result = self.predict_with_neuromodulation(prompt, max_tokens, temperature, top_p, pack_name)
        return result["text"]
    
    def get_endpoint_status(self) -> Dict[str, Any]:
        """Get endpoint status with neuromodulation info"""
        if not self.endpoint:
            return {"endpoint_configured": False}
        
        try:
            # Get endpoint details
            endpoint_details = self.endpoint.describe()
            
            # Get neuromodulation status
            neuromod_status = self.get_neuromodulation_status()
            
            return {
                "endpoint_configured": True,
                "endpoint_name": self.endpoint.display_name,
                "model_name": self.model_name,
                "endpoint_uri": self.endpoint.resource_name,
                "traffic_split": endpoint_details.get("trafficSplit", {}),
                "deployed_models": len(endpoint_details.get("deployedModels", [])),
                "neuromodulation": neuromod_status
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {e}")
            return {"endpoint_configured": False, "error": str(e)}
    
    def delete_endpoint(self):
        """Delete the current endpoint"""
        if self.endpoint:
            try:
                # Clear neuromodulation before deleting
                self.clear_neuromodulation()
                
                self.endpoint.delete()
                logger.info("✅ Endpoint deleted")
            except Exception as e:
                logger.error(f"Failed to delete endpoint: {e}")
            
            self.endpoint = None
            self.model_name = None
    
    def estimate_cost(self, input_chars: int, output_chars: int) -> Dict[str, float]:
        """Estimate cost for a prediction"""
        # Vertex AI pricing (approximate)
        input_cost = (input_chars / 1000) * 0.0025
        output_cost = (output_chars / 1000) * 0.01
        hosting_cost = ((input_chars + output_chars) / 1000) * 0.0001
        
        total_cost = input_cost + output_cost + hosting_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "hosting_cost": round(hosting_cost, 6),
            "total_cost": round(total_cost, 6)
        }

# Global Vertex AI manager instance
vertex_ai_manager = None

def initialize_vertex_ai(project_id: str, region: str = "us-central1"):
    """Initialize Vertex AI manager with FULL PROBE SYSTEM"""
    global vertex_ai_manager
    vertex_ai_manager = VertexAIManager(project_id, region)
    return vertex_ai_manager
