"""
Neuromodulation API Server
Cloud-native FastAPI service for exposing neuromodulation capabilities
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
import logging
import time
from contextlib import asynccontextmanager

# Import neuromodulation components
from neuromod import NeuromodTool
from neuromod.effects import EffectRegistry
from model_manager import model_manager
from vertex_ai_manager import initialize_vertex_ai, vertex_ai_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
neuromod_tool = None
current_pack = None

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Chat history")
    pack_name: Optional[str] = Field(None, description="Pack to apply")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    pack_applied: Optional[str] = Field(None, description="Pack that was applied")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")

class PackInfo(BaseModel):
    name: str = Field(..., description="Pack name")
    description: str = Field(..., description="Pack description")
    effects: List[Dict[str, Any]] = Field(..., description="List of effects in pack")

class EffectInfo(BaseModel):
    name: str = Field(..., description="Effect name")
    category: str = Field(..., description="Effect category")
    description: str = Field(..., description="Effect description")
    parameters: Dict[str, Any] = Field(..., description="Available parameters")

class EffectRequest(BaseModel):
    effect_name: str = Field(..., description="Name of effect to apply")
    weight: float = Field(0.5, description="Effect weight (0.0-1.0)")
    direction: str = Field("up", description="Direction: 'up', 'down', or 'neutral'")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Effect-specific parameters")

class StatusResponse(BaseModel):
    model_loaded: bool = Field(..., description="Whether model is loaded")
    current_pack: Optional[str] = Field(None, description="Currently applied pack")
    available_packs: int = Field(..., description="Number of available packs")
    available_effects: int = Field(..., description="Number of available effects")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global neuromod_tool
    
    # Startup
    logger.info("Starting Neuromodulation API Server...")
    neuromod_tool = NeuromodTool()
    
    # Initialize Vertex AI if configured
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project_id:
        try:
            initialize_vertex_ai(project_id)
            logger.info("Vertex AI initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Vertex AI: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Neuromodulation API Server...")
    if neuromod_tool:
        neuromod_tool.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Neuromodulation API",
    description="Cloud-native API for applying neuromodulation effects to language models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "neuromodulation-api"
    }

# Status endpoint
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get current system status"""
    global neuromod_tool, current_pack
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    # Get available packs and effects
    packs = neuromod_tool.list_packs()
    effect_registry = EffectRegistry()
    effects = effect_registry.list_effects()
    
    # Get model status
    model_status = model_manager.get_model_status()
    
    return StatusResponse(
        model_loaded=model_status["model_loaded"],
        current_pack=current_pack,
        available_packs=len(packs),
        available_effects=len(effects)
    )

# Pack management endpoints
@app.get("/packs", response_model=List[PackInfo])
async def list_packs():
    """List all available packs"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    packs = neuromod_tool.list_packs()
    pack_info_list = []
    
    for pack_name in packs:
        try:
            pack_config = neuromod_tool.get_pack_config(pack_name)
            pack_info_list.append(PackInfo(
                name=pack_name,
                description=pack_config.get("description", ""),
                effects=pack_config.get("effects", [])
            ))
        except Exception as e:
            logger.warning(f"Failed to get info for pack {pack_name}: {e}")
            pack_info_list.append(PackInfo(
                name=pack_name,
                description="Pack information unavailable",
                effects=[]
            ))
    
    return pack_info_list

@app.get("/packs/{pack_name}", response_model=PackInfo)
async def get_pack_info(pack_name: str):
    """Get detailed information about a specific pack"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    try:
        pack_config = neuromod_tool.get_pack_config(pack_name)
        return PackInfo(
            name=pack_name,
            description=pack_config.get("description", ""),
            effects=pack_config.get("effects", [])
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Pack '{pack_name}' not found: {str(e)}")

@app.post("/packs/{pack_name}/apply")
async def apply_pack(pack_name: str):
    """Apply a pack to the model"""
    global neuromod_tool, current_pack
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    try:
        neuromod_tool.load_pack(pack_name)
        current_pack = pack_name
        logger.info(f"Applied pack: {pack_name}")
        return {"message": f"Pack '{pack_name}' applied successfully", "pack": pack_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply pack '{pack_name}': {str(e)}")

@app.post("/packs/clear")
async def clear_pack():
    """Clear currently applied pack"""
    global neuromod_tool, current_pack
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    try:
        neuromod_tool.clear_pack()
        current_pack = None
        logger.info("Cleared current pack")
        return {"message": "Pack cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear pack: {str(e)}")

# Effect management endpoints
@app.get("/effects", response_model=List[EffectInfo])
async def list_effects():
    """List all available effects"""
    effect_registry = EffectRegistry()
    effects = effect_registry.list_effects()
    
    effect_info_list = []
    for effect_name in effects:
        # Get effect class to extract information
        effect_class = effect_registry.effects.get(effect_name)
        if effect_class:
            # Create a temporary instance to get docstring
            try:
                temp_effect = effect_class()
                doc = temp_effect.__class__.__doc__ or "No description available"
                
                # Determine category based on class name
                category = "Unknown"
                if "Sampler" in effect_name or "Temperature" in effect_name:
                    category = "Sampler"
                elif "Attention" in effect_name:
                    category = "Attention"
                elif "Memory" in effect_name or "KV" in effect_name:
                    category = "Memory"
                elif "Steering" in effect_name:
                    category = "Steering"
                elif "Activation" in effect_name:
                    category = "Activation"
                elif "Router" in effect_name or "Expert" in effect_name:
                    category = "MoE/Router"
                elif "Objective" in effect_name or "Verifier" in effect_name:
                    category = "Objective"
                elif "Input" in effect_name or "Jitter" in effect_name:
                    category = "Input"
                
                effect_info_list.append(EffectInfo(
                    name=effect_name,
                    category=category,
                    description=doc.strip(),
                    parameters={"weight": "float", "direction": "str"}
                ))
            except Exception as e:
                logger.warning(f"Failed to get info for effect {effect_name}: {e}")
                effect_info_list.append(EffectInfo(
                    name=effect_name,
                    category="Unknown",
                    description="Effect information unavailable",
                    parameters={}
                ))
    
    return effect_info_list

@app.post("/effects/apply")
async def apply_effect(effect_request: EffectRequest):
    """Apply an individual effect to the model"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    try:
        # Create effect instance
        effect_registry = EffectRegistry()
        effect = effect_registry.get_effect(
            effect_request.effect_name,
            weight=effect_request.weight,
            direction=effect_request.direction,
            **effect_request.parameters
        )
        
        # Apply effect
        neuromod_tool.apply_effect(effect)
        logger.info(f"Applied effect: {effect_request.effect_name}")
        
        return {
            "message": f"Effect '{effect_request.effect_name}' applied successfully",
            "effect": effect_request.effect_name,
            "weight": effect_request.weight,
            "direction": effect_request.direction
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply effect: {str(e)}")

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a chat response with optional neuromodulation effects"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded. Please load a model first.")
    
    try:
        # Apply pack if specified
        if request.pack_name:
            neuromod_tool.load_pack(request.pack_name)
        
        # Convert messages to prompt
        prompt = ""
        for message in request.messages:
            if message.role == "user":
                prompt += f"User: {message.content}\n"
            elif message.role == "assistant":
                prompt += f"Assistant: {message.content}\n"
        
        # Add final user message if not present
        if not prompt.endswith("Assistant:"):
            prompt += "Assistant:"
        
        # Generate response
        start_time = time.time()
        response = model_manager.generate_text(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        generation_time = time.time() - start_time
        
        # Estimate token count (rough approximation)
        tokens_generated = len(response.split()) * 1.3  # Rough estimate
        
        return ChatResponse(
            response=response,
            pack_applied=request.pack_name,
            tokens_generated=int(tokens_generated),
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/generate")
async def generate_text(
    prompt: str,
    pack_name: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0
):
    """Generate text with optional neuromodulation effects (simplified interface)"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    # Check if local model is loaded
    if model_manager.model:
        try:
            # Apply pack if specified
            if pack_name:
                neuromod_tool.load_pack(pack_name)
            
            # Generate response
            start_time = time.time()
            response = model_manager.generate_text(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            generation_time = time.time() - start_time
            
            return {
                "generated_text": response,
                "pack_applied": pack_name,
                "generation_time": generation_time,
                "prompt": prompt,
                "model_type": "local"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    # Check if Vertex AI endpoint is available
    elif vertex_ai_manager and vertex_ai_manager.endpoint:
        try:
            # Generate response using Vertex AI
            start_time = time.time()
            response = vertex_ai_manager.predict(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                pack_name=pack_name
            )
            generation_time = time.time() - start_time
            
            return {
                "generated_text": response,
                "pack_applied": pack_name,
                "generation_time": generation_time,
                "prompt": prompt,
                "model_type": "vertex_ai"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vertex AI generation failed: {str(e)}")
    
    else:
        raise HTTPException(status_code=503, detail="No model loaded. Please load a model first.")

# Model management endpoints
@app.get("/models")
async def list_models():
    """List all compatible models"""
    models = model_manager.list_compatible_models()
    
    # Add Vertex AI models if available
    if vertex_ai_manager:
        vertex_models = vertex_ai_manager.list_vertex_models()
        models.extend(vertex_models)
    
    return {"models": models}

@app.get("/models/{model_name}")
async def get_model_info(model_name: str):
    """Get information about a specific model"""
    # Check local models first
    model_info = model_manager.get_model_info(model_name)
    if model_info:
        return model_info
    
    # Check Vertex AI models
    if vertex_ai_manager:
        model_info = vertex_ai_manager.get_model_info(model_name)
        if model_info:
            return model_info
    
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

@app.post("/model/load")
async def load_model(model_name: str = "microsoft/DialoGPT-medium"):
    """Load a model (local or Vertex AI)"""
    global neuromod_tool
    
    if not neuromod_tool:
        raise HTTPException(status_code=503, detail="Neuromodulation tool not initialized")
    
    try:
        # Try local model first
        if model_manager.is_model_compatible(model_name):
            success = model_manager.load_model(model_name)
            if success:
                logger.info(f"Local model loaded: {model_name}")
                memory_info = model_manager.estimate_memory_usage()
                return {
                    "message": f"Local model '{model_name}' loaded successfully",
                    "model": model_name,
                    "memory_usage": memory_info,
                    "type": "local"
                }
        
        # Try Vertex AI model
        if vertex_ai_manager and vertex_ai_manager.get_model_info(model_name):
            endpoint_name = f"neuromod-{model_name.replace('/', '-').replace('_', '-')}"
            endpoint_id = vertex_ai_manager.create_custom_endpoint(model_name, endpoint_name)
            logger.info(f"Vertex AI endpoint created: {endpoint_id}")
            return {
                "message": f"Vertex AI model '{model_name}' endpoint created",
                "model": model_name,
                "endpoint_id": endpoint_id,
                "type": "vertex_ai"
            }
        
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/model/unload")
async def unload_model():
    """Unload current model"""
    try:
        model_manager.unload_model()
        return {"message": "Model unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    """Get model loading status"""
    status = model_manager.get_model_status()
    memory_info = model_manager.estimate_memory_usage()
    return {**status, "memory_usage": memory_info}

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
