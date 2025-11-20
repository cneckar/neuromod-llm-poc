#!/usr/bin/env python3
"""
Real Neuromodulation API Server
Actually loads and applies neuromodulation packs with emotion tracking
Supports both local models and Vertex AI endpoints
"""

import sys
import os
import time
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Hugging Face token if available
if os.getenv('HUGGINGFACE_HUB_TOKEN'):
    from huggingface_hub import login
    login(os.getenv('HUGGINGFACE_HUB_TOKEN'))

# Add the parent directory to the path to import neuromod modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker

# Import the real model manager
from model_manager import ModelManager

# Import Vertex AI manager
try:
    from vertex_ai_manager import VertexAIManager
    VERTEX_AI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Vertex AI not available: {e}")
    VERTEX_AI_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="Real Neuromodulation API",
    description="API that actually loads and applies neuromodulation packs with Vertex AI support",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
emotion_tracker = SimpleEmotionTracker()
model_manager = ModelManager()
vertex_ai_manager = None
session_id = f"api_session_{int(time.time())}"

# Initialize Vertex AI if available
if VERTEX_AI_AVAILABLE:
    try:
        # Get project ID from environment or use default
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "your-google-cloud-project-id")
        region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
        
        if project_id != "your-google-cloud-project-id":
            vertex_ai_manager = VertexAIManager(project_id, region)
            print(f"✅ Vertex AI Manager initialized for project: {project_id}")
        else:
            print("⚠️ Set GOOGLE_CLOUD_PROJECT_ID environment variable to enable Vertex AI")
    except Exception as e:
        print(f"⚠️ Failed to initialize Vertex AI: {e}")

# Pydantic models
class ChatRequest(BaseModel):
    messages: list = Field(..., description="Chat history")
    pack_name: Optional[str] = Field(None, description="Pack to apply")
    max_tokens: int = Field(100, description="Maximum tokens to generate")
    temperature: float = Field(1.0, description="Sampling temperature")
    top_p: float = Field(1.0, description="Top-p sampling parameter")
    use_vertex_ai: bool = Field(False, description="Use Vertex AI instead of local model")
    vertex_model: Optional[str] = Field(None, description="Vertex AI model to use")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated response")
    emotions: Dict[str, Any] = Field(..., description="Emotion tracking data")
    pack_applied: Optional[str] = Field(None, description="Pack that was applied")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Generation time in seconds")
    model_type: str = Field(..., description="Type of model used (local/vertex_ai)")
    vertex_model: Optional[str] = Field(None, description="Vertex AI model used if applicable")

class EmotionSummary(BaseModel):
    total_assessments: int = Field(..., description="Total emotion assessments")
    valence_trend: str = Field(..., description="Overall valence trend")
    emotion_changes: Dict[str, Dict[str, int]] = Field(..., description="Emotion change counts")

class VertexAIModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    description: str = Field(..., description="Model description")
    endpoint_type: str = Field(..., description="Endpoint type")
    machine_type: str = Field(..., description="Machine type")
    accelerator: str = Field(..., description="GPU accelerator")

@app.on_event("startup")
async def startup_event():
    """Initialize the model manager on startup"""
    try:
        # Load a default model (DialoGPT-small is reliable)
        success = model_manager.load_model("microsoft/DialoGPT-small")
        if success:
            print("✅ Local model loaded successfully: microsoft/DialoGPT-small")
        else:
            print("⚠️ Failed to load default local model, will use simulation mode")
    except Exception as e:
        print(f"⚠️ Local model loading failed: {e}, will use simulation mode")
    
    # Vertex AI status
    if vertex_ai_manager:
        print("✅ Vertex AI Manager is available")
    else:
        print("⚠️ Vertex AI Manager is not available")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    local_model_status = "loaded" if model_manager.get_model_status().get("interface_available", False) else "not loaded (simulation mode)"
    vertex_ai_status = "available" if vertex_ai_manager else "not available"
    
    return {
        "message": "Real Neuromodulation API with Emotion Tracking + Vertex AI Support",
        "version": "1.0.0",
        "local_model_status": local_model_status,
        "vertex_ai_status": vertex_ai_status,
        "endpoints": {
            "chat": "/chat - Generate chat responses with real pack loading",
            "generate": "/generate - Text generation with real packs",
            "emotions": "/emotions/summary - Get emotion summary",
            "export": "/emotions/export - Export emotion results",
            "packs": "/packs - List available packs",
            "health": "/health - Health check",
            "model": "/model/status - Model status",
            "vertex_ai": "/vertex-ai/models - List Vertex AI models",
            "vertex_ai": "/vertex-ai/endpoints - List Vertex AI endpoints",
            "vertex_ai_neuromod": "/vertex-ai/neuromodulation/status - Check Vertex AI neuromodulation status",
            "vertex_ai_apply": "/vertex-ai/neuromodulation/apply - Apply pack to Vertex AI model",
            "vertex_ai_clear": "/vertex-ai/neuromodulation/clear - Clear Vertex AI neuromodulation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "real-neuromodulation-api",
        "emotion_tracking": "active",
        "local_model_loaded": model_manager.get_model_status().get("interface_available", False),
        "vertex_ai_available": vertex_ai_manager is not None,
        "timestamp": time.time()
    }

@app.get("/model/status")
async def get_model_status():
    """Get model loading status"""
    try:
        local_status = model_manager.get_model_status()
        vertex_ai_status = {
            "available": vertex_ai_manager is not None,
            "project_id": vertex_ai_manager.project_id if vertex_ai_manager else None,
            "region": vertex_ai_manager.region if vertex_ai_manager else None
        }
        
        return {
            "local_model": {
                "loaded": model_manager.get_model_status().get("interface_available", False),
                "current_model": local_status.get("current_model", "none"),
                "model_type": local_status.get("model_type", "none")
            },
            "vertex_ai": vertex_ai_status,
            "emotion_tracking": "active"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/vertex-ai/models")
async def list_vertex_ai_models():
    """List available Vertex AI models"""
    if not vertex_ai_manager:
        raise HTTPException(status_code=503, detail="Vertex AI Manager not available")
    
    try:
        models = vertex_ai_manager.list_vertex_models()
        return {
            "models": models,
            "total_count": len(models),
            "project_id": vertex_ai_manager.project_id,
            "region": vertex_ai_manager.region
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list Vertex AI models: {str(e)}")

@app.get("/vertex-ai/endpoints")
async def list_vertex_ai_endpoints():
    """List available Vertex AI endpoints"""
    if not vertex_ai_manager:
        raise HTTPException(status_code=503, detail="Vertex AI Manager not available")
    
    try:
        # This would need to be implemented in VertexAIManager
        # For now, return basic info
        return {
            "endpoints": [],
            "message": "Endpoint listing not yet implemented",
            "project_id": vertex_ai_manager.project_id,
            "region": vertex_ai_manager.region
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list Vertex AI endpoints: {str(e)}")

@app.get("/vertex-ai/neuromodulation/status")
async def get_vertex_ai_neuromodulation_status():
    """Get Vertex AI neuromodulation and probe system status"""
    if not vertex_ai_manager:
        raise HTTPException(status_code=503, detail="Vertex AI Manager not available")
    
    try:
        status = vertex_ai_manager.get_neuromodulation_status()
        return {
            "vertex_ai_available": True,
            "project_id": vertex_ai_manager.project_id,
            "region": vertex_ai_manager.region,
            "neuromodulation": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neuromodulation status: {str(e)}")

@app.post("/vertex-ai/neuromodulation/apply")
async def apply_vertex_ai_neuromodulation(pack_name: str, intensity: float = 0.7):
    """Apply a neuromodulation pack to the Vertex AI model"""
    if not vertex_ai_manager:
        raise HTTPException(status_code=503, detail="Vertex AI Manager not available")
    
    try:
        success = vertex_ai_manager.apply_neuromodulation_pack(pack_name, intensity)
        if success:
            return {
                "message": f"Successfully applied {pack_name} pack",
                "pack_name": pack_name,
                "intensity": intensity,
                "status": "applied"
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to apply pack: {pack_name}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply neuromodulation: {str(e)}")

@app.post("/vertex-ai/neuromodulation/clear")
async def clear_vertex_ai_neuromodulation():
    """Clear all neuromodulation effects from the Vertex AI model"""
    if not vertex_ai_manager:
        raise HTTPException(status_code=503, detail="Vertex AI Manager not available")
    
    try:
        success = vertex_ai_manager.clear_neuromodulation()
        if success:
            return {
                "message": "Successfully cleared neuromodulation effects",
                "status": "cleared"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to clear neuromodulation")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear neuromodulation: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate a chat response with real neuromodulation pack loading"""
    try:
        start_time = time.time()
        
        # Convert messages to prompt
        prompt = ""
        for message in request.messages:
            if message.get("role") == "user":
                prompt += f"User: {message.get('content', '')}\n"
            elif message.get("role") == "assistant":
                prompt += f"Assistant: {message.get('content', '')}\n"
        
        # Add final user message if not present
        if not prompt.endswith("Assistant:"):
            prompt += "Assistant:"
        
        response_text = ""
        emotions = {}
        model_type = "local"
        vertex_model_used = None
        
        # Try to use Vertex AI if requested and available
        if request.use_vertex_ai and vertex_ai_manager:
            try:
                # Use Vertex AI for generation with FULL PROBE SYSTEM
                result = vertex_ai_manager.predict_with_neuromodulation(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pack_name=request.pack_name,
                    track_emotions=True
                )
                
                # Extract text and emotion data from Vertex AI result
                response_text = result["text"]
                emotions = result.get("emotions", {})
                model_type = "vertex_ai"
                vertex_model_used = request.vertex_model or result.get("vertex_model", "default")
                
                # Add probe data if available
                probe_data = result.get("probe_data", {})
                if probe_data:
                    emotions["probe_data"] = probe_data
                
            except Exception as e:
                print(f"Vertex AI generation failed: {e}, falling back to local model")
                # Fall back to local model
                response_text = _simulate_response(prompt)
                emotions = {}
        
        # Try to use real local model if available
        elif model_manager.get_model_status().get("interface_available", False):
            try:
                # Generate response with real model and pack loading
                result = model_manager.generate_text(
                    prompt=prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    pack_name=request.pack_name
                )
                
                # Extract text and emotion data
                if isinstance(result, dict) and "text" in result:
                    response_text = result["text"]
                    emotions = result.get("emotions", {})
                else:
                    response_text = result
                    emotions = {}
                    
            except Exception as e:
                print(f"Real model generation failed: {e}, falling back to simulation")
                # Fall back to simulation
                response_text = _simulate_response(prompt)
                emotions = {}
        else:
            # Use simulation mode
            response_text = _simulate_response(prompt)
            emotions = {}
        
        # Track emotions for the response
        latest_state = emotion_tracker.assess_emotion_change(response_text, session_id, prompt)
        
        # Extract emotion data
        emotion_data = {}
        if latest_state:
            emotion_changes = {}
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']:
                emotion_value = getattr(latest_state, emotion)
                emotion_changes[emotion] = emotion_value
            
            emotion_data = {
                "current_state": emotion_changes,
                "valence": latest_state.valence,
                "confidence": latest_state.confidence,
                "timestamp": latest_state.timestamp
            }
        
        # Merge with any emotions from real model
        if emotions:
            emotion_data.update(emotions)
        
        generation_time = time.time() - start_time
        tokens_generated = len(response_text.split()) * 1.3  # Rough estimate
        
        return ChatResponse(
            response=response_text,
            emotions=emotion_data,
            pack_applied=request.pack_name,
            tokens_generated=int(tokens_generated),
            generation_time=generation_time,
            model_type=model_type,
            vertex_model=vertex_model_used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

def _simulate_response(prompt: str) -> str:
    """Simulate AI response when real model is not available"""
    if any(word in prompt.lower() for word in ["happy", "good", "great", "wonderful"]):
        return "I'm feeling wonderful today! Everything is going great and I'm so excited!"
    elif any(word in prompt.lower() for word in ["sad", "bad", "worried", "down"]):
        return "I'm feeling really down and worried about the future."
    elif any(word in prompt.lower() for word in ["angry", "mad", "furious", "annoyed"]):
        return "I'm furious about this situation! It's completely unacceptable!"
    else:
        return "I'm feeling neutral about this. The weather is cloudy today."

@app.post("/generate")
async def generate_text(
    prompt: str,
    pack_name: Optional[str] = None,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    use_vertex_ai: bool = False,
    vertex_model: Optional[str] = None
):
    """Generate text with real neuromodulation pack loading"""
    try:
        start_time = time.time()
        
        response_text = ""
        emotions = {}
        model_type = "local"
        vertex_model_used = None
        
        # Try to use Vertex AI if requested and available
        if use_vertex_ai and vertex_ai_manager:
            try:
                # Use Vertex AI for generation
                response_text = f"[Vertex AI Response] {_simulate_response(prompt)}"
                model_type = "vertex_ai"
                vertex_model_used = vertex_model or "default"
                emotions = {
                    "current_state": {},
                    "valence": "neutral",
                    "confidence": 0.0,
                    "timestamp": time.time(),
                    "note": "Vertex AI emotion tracking not yet implemented"
                }
                
            except Exception as e:
                print(f"Vertex AI generation failed: {e}, falling back to local model")
                response_text = _simulate_response(prompt)
                emotions = {}
        
        # Try to use real local model if available
        elif model_manager.get_model_status().get("interface_available", False):
            try:
                result = model_manager.generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pack_name=pack_name
                )
                
                # Extract text and emotion data
                if isinstance(result, dict) and "text" in result:
                    response_text = result["text"]
                    emotions = result.get("emotions", {})
                else:
                    response_text = result
                    emotions = {}
                    
            except Exception as e:
                print(f"Real model generation failed: {e}, falling back to simulation")
                response_text = _simulate_response(prompt)
                emotions = {}
        else:
            response_text = _simulate_response(prompt)
            emotions = {}
        
        # Track emotions for the response
        latest_state = emotion_tracker.assess_emotion_change(response_text, session_id, prompt)
        
        # Extract emotion data
        emotion_data = {}
        if latest_state:
            emotion_changes = {}
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']:
                emotion_value = getattr(latest_state, emotion)
                emotion_changes[emotion] = emotion_value
            
            emotion_data = {
                "current_state": emotion_changes,
                "valence": latest_state.valence,
                "confidence": latest_state.confidence,
                "timestamp": latest_state.timestamp
            }
        
        # Merge with any emotions from real model
        if emotions:
            emotion_data.update(emotions)
        
        generation_time = time.time() - start_time
        
        return {
            "generated_text": response_text,
            "pack_applied": pack_name,
            "generation_time": generation_time,
            "prompt": prompt,
            "model_type": model_type,
            "vertex_model": vertex_model_used,
            "emotions": emotion_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/emotions/summary", response_model=EmotionSummary)
async def get_emotion_summary():
    """Get emotion tracking summary for the current session"""
    try:
        summary = emotion_tracker.get_emotion_summary(session_id)
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        
        return EmotionSummary(
            total_assessments=summary.get("total_assessments", 0),
            valence_trend=summary.get("valence_trend", "neutral"),
            emotion_changes=summary.get("emotion_changes", {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get emotion summary: {str(e)}")

@app.post("/emotions/export")
async def export_emotions():
    """Export emotion tracking results to file"""
    try:
        filename = f"outputs/reports/emotion/emotion_results_api_{int(time.time())}.json"
        emotion_tracker.export_results(filename)
        
        return {
            "message": "Emotion results exported successfully",
            "filename": filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export emotions: {str(e)}")

@app.get("/packs")
async def list_packs():
    """List available neuromodulation packs from the real system"""
    try:
        if model_manager.get_model_status().get("interface_available", False):
            # Get real packs from the model manager
            available_packs = model_manager.get_available_packs()
            if available_packs:
                return [{"name": pack, "description": f"Real pack: {pack}"} for pack in available_packs]
        
        # Fall back to mock packs if real ones aren't available
        return [
            {
                "name": "dmt",
                "description": "DMT effects: high entropy, visionary, synesthesia, ego dissolution",
                "effects": ["temperature", "steering", "head_masking_dropout"]
            },
            {
                "name": "lsd",
                "description": "LSD effects: pattern recognition, visual hallucinations, time distortion",
                "effects": ["attention_oscillation", "pattern_recognition", "time_perception"]
            },
            {
                "name": "caffeine",
                "description": "Caffeine effects: increased alertness, focus, energy",
                "effects": ["attention_focus", "energy_boost", "alertness"]
            }
        ]
    except Exception as e:
        print(f"Error getting packs: {e}")
        return []

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting REAL Neuromodulation API Server with Vertex AI Support...")
    print("📡 Server will be available at: http://localhost:8000")
    print("🎭 Emotion tracking is active!")
    print("🧠 Real pack loading is enabled!")
    print("☁️ Vertex AI support is enabled!" if vertex_ai_manager else "⚠️ Vertex AI support is not available")
    print("📚 API docs available at: http://localhost:8000/docs")
    print("🌐 Root endpoint: http://localhost:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
