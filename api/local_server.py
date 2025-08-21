#!/usr/bin/env python3
"""
Local Neuromodulation API Server
Simple FastAPI server for testing neuromodulation with local models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import time
import os
import uvicorn

# Import the enhanced model manager
from model_manager import EnhancedModelManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate from")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    pack_name: Optional[str] = Field(None, description="Neuromodulation pack to apply")
    custom_pack: Optional[Dict[str, Any]] = Field(None, description="Custom pack definition")
    individual_effects: Optional[List[Dict[str, Any]]] = Field(None, description="Individual effects to apply")
    multiple_packs: Optional[List[str]] = Field(None, description="Multiple packs to combine")

class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text")
    pack_applied: Optional[str] = Field(None, description="Pack that was applied")
    custom_pack_applied: Optional[str] = Field(None, description="Custom pack that was applied")
    individual_effects_applied: Optional[List[str]] = Field(None, description="Individual effects applied")
    multiple_packs_applied: Optional[List[str]] = Field(None, description="Multiple packs applied")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class StatusResponse(BaseModel):
    interface_type: Optional[str] = Field(None, description="Current interface type")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_name: Optional[str] = Field(None, description="Current model name")
    neuromodulation_available: bool = Field(..., description="Whether neuromodulation is available")
    available_packs: int = Field(..., description="Number of available packs")
    available_effects: int = Field(..., description="Number of available effects")



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    # Startup
    logger.info("Starting Local Neuromodulation API Server...")
    logger.info("Use /load-model/{model_name} to load a model")
    logger.info("Use /status to check current status")
    
    # Auto-load T5-small as the default working model
    try:
        logger.info("Auto-loading T5-small as default model...")
        model_manager.load_local_model("t5-small")
        logger.info("✅ T5-small loaded successfully as default model")
    except Exception as e:
        logger.warning(f"Could not auto-load T5-small: {e}")
        logger.info("You can manually load a model using /load-model/{model_name}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Local Neuromodulation API Server...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Local Neuromodulation API",
    description="Local API for testing neuromodulation with local models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the enhanced model manager
model_manager = EnhancedModelManager()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Local Neuromodulation API Server", "status": "running"}

@app.post("/load-model/{model_name}")
async def load_model(model_name: str):
    """Load a local model"""
    try:
        logger.info(f"Loading model: {model_name}")
        model_manager.load_local_model(model_name)
        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "status": model_manager.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: Dict[str, Any]):
    """Generate text using the loaded model"""
    try:
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 50)
        temperature = request.get("temperature", 1.0)
        top_p = request.get("top_p", 1.0)
        
        # Neuromodulation parameters
        pack_name = request.get("pack_name")
        custom_pack = request.get("custom_pack")
        individual_effects = request.get("individual_effects")
        multiple_packs = request.get("multiple_packs")
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Generate text with optional neuromodulation
        generated_text = model_manager.generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pack_name=pack_name,
            custom_pack=custom_pack,
            individual_effects=individual_effects,
            multiple_packs=multiple_packs
        )
        
        return {
            "success": True,
            "generated_text": generated_text,
            "prompt": prompt,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
        }
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get current server and model status"""
    return model_manager.get_status()

@app.get("/available-packs")
async def get_available_packs():
    """Get available neuromodulation packs"""
    try:
        packs = model_manager.get_available_packs()
        return {
            "success": True,
            "available_packs": packs,
            "count": len(packs)
        }
    except Exception as e:
        logger.error(f"Failed to get available packs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-effects")
async def get_available_effects():
    """Get available neuromodulation effects"""
    try:
        effects = model_manager.get_available_effects()
        return {
            "success": True,
            "available_effects": effects,
            "count": len(effects)
        }
    except Exception as e:
        logger.error(f"Failed to get available effects: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/connect-vertex-ai")
async def connect_vertex_ai(request: Dict[str, Any]):
    """Connect to a Vertex AI endpoint"""
    try:
        endpoint_name = request.get("endpoint_name")
        project_id = request.get("project_id")
        location = request.get("location", "us-central1")
        
        if not endpoint_name or not project_id:
            raise HTTPException(status_code=400, detail="endpoint_name and project_id are required")
        
        model_manager.connect_vertex_ai(endpoint_name, project_id, location)
        return {
            "success": True,
            "message": f"Connected to Vertex AI endpoint: {endpoint_name}",
            "status": model_manager.get_status()
        }
        
    except Exception as e:
        logger.error(f"Failed to connect to Vertex AI: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload")
async def unload_model():
    """Unload the current model"""
    try:
        model_manager.unload_current_interface()
        return {
            "success": True,
            "message": "Model unloaded successfully",
            "status": model_manager.get_status()
        }
    except Exception as e:
        logger.error(f"Failed to unload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
