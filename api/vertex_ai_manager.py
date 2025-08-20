"""
Vertex AI Manager for Pay-Per-Use Model Serving
Integrates with Google Cloud Vertex AI for Llama 3 and other models
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
import requests
from google.cloud import aiplatform
from google.auth import default

logger = logging.getLogger(__name__)

class VertexAIManager:
    """Manages Vertex AI endpoints for pay-per-use model serving"""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        self.endpoint = None
        self.model_name = None
        
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
                "accelerator": "NVIDIA_TESLA_T4"
            },
            "meta-llama/Meta-Llama-3.1-70B": {
                "type": "causal",
                "description": "Llama 3.1 70B - High quality, higher cost",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-16",
                "accelerator": "NVIDIA_TESLA_A100"
            },
            
            # Qwen series
            "Qwen/Qwen2.5-7B": {
                "type": "causal",
                "description": "Qwen 2.5 7B - Good performance, reasonable cost",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-4",
                "accelerator": "NVIDIA_TESLA_T4"
            },
            "Qwen/Qwen2.5-32B": {
                "type": "causal",
                "description": "Qwen 2.5 32B - High quality",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-16",
                "accelerator": "NVIDIA_TESLA_A100"
            },
            
            # Mixtral series
            "mistralai/Mixtral-8x7B-v0.1": {
                "type": "causal",
                "description": "Mixtral 8x7B - MoE model, good quality",
                "endpoint_type": "text-generation",
                "machine_type": "n1-standard-8",
                "accelerator": "NVIDIA_TESLA_V100"
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
                "accelerator": info["accelerator"]
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
    
    def create_custom_endpoint(self, model_name: str, endpoint_name: str = None) -> str:
        """Create a custom endpoint with neuromodulation support"""
        try:
            if endpoint_name is None:
                endpoint_name = f"neuromod-{model_name.replace('/', '-').replace('_', '-')}"
            
            model_info = self.get_model_info(model_name)
            if not model_info:
                raise ValueError(f"Model {model_name} not supported")
            
            logger.info(f"Creating custom endpoint: {endpoint_name}")
            
            # Create custom container for neuromodulation
            container_uri = self._build_custom_container(model_name)
            
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_name,
                project=self.project_id,
                location=self.region
            )
            
            # Deploy model to endpoint
            model = aiplatform.Model.upload(
                display_name=f"{model_name}-neuromod",
                container_spec={
                    "image_uri": container_uri,
                    "predict_route": "/predict",
                    "health_route": "/health"
                },
                serving_container_environment_variables={
                    "MODEL_NAME": model_name,
                    "PROJECT_ID": self.project_id
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
            
            logger.info(f"Endpoint created: {endpoint.resource_name}")
            return endpoint.resource_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint: {e}")
            raise
    
    def _build_custom_container(self, model_name: str) -> str:
        """Build custom container with neuromodulation support"""
        # This would build and push a custom container
        # For now, we'll use a placeholder
        return f"gcr.io/{self.project_id}/neuromod-{model_name.replace('/', '-')}:latest"
    
    def predict(self, prompt: str, max_tokens: int = 100, 
               temperature: float = 1.0, top_p: float = 1.0,
               pack_name: str = None) -> str:
        """Generate text using Vertex AI endpoint"""
        if not self.endpoint:
            raise RuntimeError("No endpoint configured")
        
        try:
            # Prepare prediction request
            prediction_request = {
                "instances": [{
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "pack_name": pack_name
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
                
                logger.info(f"Prediction completed in {prediction_time:.2f}s")
                return generated_text
            else:
                raise RuntimeError("No prediction returned")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def get_endpoint_status(self) -> Dict[str, Any]:
        """Get endpoint status"""
        if not self.endpoint:
            return {"endpoint_configured": False}
        
        try:
            # Get endpoint details
            endpoint_details = self.endpoint.describe()
            
            return {
                "endpoint_configured": True,
                "endpoint_name": self.endpoint.display_name,
                "model_name": self.model_name,
                "endpoint_uri": self.endpoint.resource_name,
                "traffic_split": endpoint_details.get("trafficSplit", {}),
                "deployed_models": len(endpoint_details.get("deployedModels", []))
            }
        except Exception as e:
            logger.error(f"Failed to get endpoint status: {e}")
            return {"endpoint_configured": False, "error": str(e)}
    
    def delete_endpoint(self):
        """Delete the current endpoint"""
        if self.endpoint:
            try:
                self.endpoint.delete()
                logger.info("Endpoint deleted")
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
    """Initialize Vertex AI manager"""
    global vertex_ai_manager
    vertex_ai_manager = VertexAIManager(project_id, region)
    return vertex_ai_manager
