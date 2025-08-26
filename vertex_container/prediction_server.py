"""
Vertex AI Custom Prediction Server
Handles model inference with FULL NEUROMODULATION PROBE SYSTEM and emotion tracking
"""

import os
import json
import logging
import time
from typing import Dict, Any, List
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage

# Import neuromodulation components
import sys
import os
sys.path.append('/app')

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import neuromodulation system with FULL PROBE SYSTEM
try:
    from neuromod import NeuromodTool
    from neuromod.effects import EffectRegistry
    from neuromod.pack_system import Pack, EffectConfig
    from neuromod.testing.simple_emotion_tracker import SimpleEmotionTracker
    NEUROMOD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neuromodulation system not available: {e}")
    NEUROMOD_AVAILABLE = False

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
neuromod_tool = None
emotion_tracker = None
model_name = None
current_pack = None
probe_hooks = []
probe_data = []

def load_model():
    """Load the model and tokenizer with FULL PROBE SYSTEM"""
    global model, tokenizer, neuromod_tool, emotion_tracker, model_name
    
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B")
    
    logger.info(f"Loading model with FULL PROBE SYSTEM: {model_name}")
    
    try:
        # Check for Hugging Face credentials
        hf_token = os.environ.get("HUGGINGFACE_TOKEN")
        hf_username = os.environ.get("HUGGINGFACE_USERNAME")
        
        if not hf_token and ("llama" in model_name.lower() or "meta-llama" in model_name.lower()):
            logger.warning("HUGGINGFACE_TOKEN not set. Llama models require authentication.")
            logger.warning("Set HUGGINGFACE_TOKEN environment variable in Vertex AI deployment.")
            logger.warning("Model loading may fail without proper authentication.")
        
        # Load tokenizer with authentication if available
        if hf_token:
            logger.info("Using Hugging Face authentication")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
        else:
            logger.info("Loading tokenizer without authentication")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization for GPU efficiency
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with authentication if available
        if hf_token:
            logger.info("Loading model with authentication")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token
            )
        else:
            logger.info("Loading model without authentication")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Initialize neuromodulation tool with FULL PROBE SYSTEM
        if NEUROMOD_AVAILABLE:
            logger.info("Initializing neuromodulation")
            from neuromod.pack_system import PackRegistry
            registry = PackRegistry()
            neuromod_tool = NeuromodTool(registry=registry, model=model, tokenizer=tokenizer)
            emotion_tracker = SimpleEmotionTracker()
            logger.info("✅ FULL PROBE SYSTEM initialized with emotion tracking")
            
            logger.info("Registering Probe Hooks")
            # Register probe hooks on the loaded model
            register_probe_hooks()
            
        else:
            neuromod_tool = None
            emotion_tracker = None
            logger.warning("❌ Neuromodulation system not available")
        
        logger.info("✅ Model loaded successfully with FULL PROBE SYSTEM")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def register_probe_hooks():
    """Register PyTorch forward hooks for REAL-TIME PROBE MONITORING"""
    global model, neuromod_tool, probe_hooks
    
    if not model or not neuromod_tool:
        return
    
    try:
        logger.info("🔌 Registering PyTorch forward hooks for probe monitoring...")
        
        # Clear any existing hooks
        remove_probe_hooks()
        
        # Register hooks on attention layers
        for name, module in model.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                hook = module.register_forward_hook(
                    lambda mod, inp, output, name=name: attention_probe_hook(mod, inp, output, name)
                )
                probe_hooks.append(hook)
                logger.info(f"🔌 Hook registered on: {name}")
            
            # Register hooks on feed-forward layers
            elif "mlp" in name.lower() or "ffn" in name.lower():
                hook = module.register_forward_hook(
                    lambda mod, inp, output, name=name: mlp_probe_hook(mod, inp, output, name)
                )
                probe_hooks.append(hook)
                logger.info(f"🔌 Hook registered on: {name}")
            
            # Register hooks on output layers
            elif "output" in name.lower() or "lm_head" in name.lower():
                hook = module.register_forward_hook(
                    lambda mod, inp, output, name=name: output_probe_hook(mod, inp, output, name)
                )
                probe_hooks.append(hook)
                logger.info(f"🔌 Hook registered on: {name}")
        
        logger.info(f"✅ Registered {len(probe_hooks)} probe hooks")
        
    except Exception as e:
        logger.error(f"Failed to register probe hooks: {e}")

def remove_probe_hooks():
    """Remove all PyTorch forward hooks"""
    global probe_hooks
    
    try:
        for hook in probe_hooks:
            hook.remove()
        probe_hooks = []
        logger.info("🔌 Removed all probe hooks")
    except Exception as e:
        logger.error(f"Failed to remove probe hooks: {e}")

def attention_probe_hook(module, input, output, name):
    """Probe hook for attention layers"""
    try:
        # Extract attention patterns
        if hasattr(output, 'last_hidden_state'):
            attention_output = output.last_hidden_state
        elif isinstance(output, tuple) and len(output) > 0:
            attention_output = output[0]
        else:
            attention_output = output
        
        # Calculate attention metrics
        if attention_output is not None and hasattr(attention_output, 'shape'):
            # Attention oscillation detection
            if attention_output.dim() >= 3:
                # Calculate attention variance across heads
                attention_variance = torch.var(attention_output, dim=-1).mean().item()
                
                probe_data.append({
                    "timestamp": time.time(),
                    "layer": name,
                    "probe_type": "attention_oscillation",
                    "attention_variance": attention_variance,
                    "attention_shape": list(attention_output.shape),
                    "effect_applied": current_pack
                })
                
    except Exception as e:
        logger.debug(f"Attention probe hook error: {e}")

def mlp_probe_hook(module, input, output, name):
    """Probe hook for MLP/feed-forward layers"""
    try:
        if output is not None and hasattr(output, 'shape'):
            # Calculate MLP activation patterns
            if output.dim() >= 2:
                activation_mean = torch.mean(output).item()
                activation_std = torch.std(output).item()
                
                probe_data.append({
                    "timestamp": time.time(),
                    "layer": name,
                    "probe_type": "mlp_activation",
                    "activation_mean": activation_mean,
                    "activation_std": activation_std,
                    "activation_shape": list(output.shape),
                    "effect_applied": current_pack
                })
                
    except Exception as e:
        logger.debug(f"MLP probe hook error: {e}")

def output_probe_hook(module, input, output, name):
    """Probe hook for output layers"""
    try:
        if output is not None and hasattr(output, 'shape'):
            # Calculate output distribution metrics
            if output.dim() >= 2:
                output_entropy = torch.softmax(output, dim=-1)
                entropy_value = -torch.sum(output_entropy * torch.log(output_entropy + 1e-8)).item()
                
                probe_data.append({
                    "timestamp": time.time(),
                    "layer": name,
                    "probe_type": "output_distribution",
                    "entropy": entropy_value,
                    "output_shape": list(output.shape),
                    "effect_applied": current_pack
                })
                
    except Exception as e:
        logger.debug(f"Output probe hook error: {e}")

def apply_neuromodulation(pack_name: str = None, custom_pack: Dict = None, 
                         individual_effects: List[Dict] = None, 
                         multiple_packs: List[str] = None) -> bool:
    """Apply neuromodulation effects with FULL PROBE SYSTEM monitoring"""
    global neuromod_tool, current_pack, probe_data
    
    if not neuromod_tool or not NEUROMOD_AVAILABLE:
        logger.warning("Neuromodulation not available")
        return False
    
    try:
        # Clear any existing effects first
        neuromod_tool.clear()
        current_pack = None
        probe_data.clear()  # Clear previous probe data
        
        # Method 1: Single predefined pack
        if pack_name:
            success = neuromod_tool.apply(pack_name, intensity=0.7)
            if success:
                current_pack = pack_name
                logger.info(f"✅ Applied predefined pack: {pack_name}")
            else:
                logger.error(f"Failed to apply pack: {pack_name}")
                return False
        
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
            neuromod_tool.pack_manager.apply_pack(pack, model)
            current_pack = pack.name
            logger.info(f"✅ Applied custom pack: {pack.name}")
        
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
                neuromod_tool.pack_manager.apply_pack(single_pack, model)
                current_pack = single_pack.name
                logger.info(f"✅ Applied individual effect: {effect_name}")
        
        # Method 4: Multiple packs (combine effects)
        elif multiple_packs:
            all_effects = []
            for pack_name in multiple_packs:
                try:
                    # Load pack to get effects
                    neuromod_tool.load_pack(pack_name)
                    # Get the active effects and add to our list
                    pack_info = neuromod_tool.get_effect_info()
                    # Note: This is a simplified approach - in practice you'd want to
                    # extract the actual effect configurations from each pack
                    logger.info(f"Loaded pack: {pack_name}")
                except Exception as e:
                    logger.warning(f"Failed to load pack {pack_name}: {e}")
            
            # Apply the combined effects
            # Note: This is a simplified implementation - you'd want more sophisticated
            # pack combination logic in practice
            current_pack = f"combined_{'_'.join(multiple_packs)}"
            logger.info(f"✅ Applied multiple packs: {multiple_packs}")
        
        # Apply effects to the model
        if any([pack_name, custom_pack, individual_effects, multiple_packs]):
            neuromod_tool.apply_to_model(model)
            logger.info(f"✅ Applied neuromodulation effects to model: {current_pack}")
            
            # Ensure probe hooks are active
            if not probe_hooks:
                register_probe_hooks()
            
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to apply neuromodulation: {e}")
        return False

def generate_text_with_probes(prompt: str, max_tokens: int = 100, 
                            temperature: float = 1.0, top_p: float = 1.0,
                            pack_name: str = None, custom_pack: Dict = None,
                            individual_effects: List[Dict] = None,
                            multiple_packs: List[str] = None,
                            track_emotions: bool = True) -> Dict[str, Any]:
    """Generate text with FULL PROBE SYSTEM monitoring and emotion tracking"""
    global model, tokenizer, neuromod_tool, emotion_tracker, current_pack, probe_data
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Clear previous probe data
        probe_data.clear()
        
        # Apply neuromodulation effects
        neuromod_applied = apply_neuromodulation(
            pack_name=pack_name,
            custom_pack=custom_pack,
            individual_effects=individual_effects,
            multiple_packs=multiple_packs
        )
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate with neuromodulation effects applied and probes active
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generation_time = time.time() - start_time
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # Track emotions if enabled
        emotion_data = {}
        if track_emotions and emotion_tracker:
            try:
                session_id = f"vertex_container_{int(time.time())}"
                latest_state = emotion_tracker.assess_emotion_change(
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
            except Exception as e:
                logger.warning(f"Emotion tracking failed: {e}")
        
        # Prepare probe data summary
        probe_summary = {
            "total_probes": len(probe_data),
            "probe_types": list(set(p['probe_type'] for p in probe_data)),
            "layers_monitored": list(set(p['layer'] for p in probe_data)),
            "effect_applied": current_pack,
            "generation_time": generation_time
        }
        
        # Return comprehensive result with probe data
        return {
            "text": generated_text,
            "probe_data": probe_data.copy(),  # Copy to preserve for analysis
            "probe_summary": probe_summary,
            "emotions": emotion_data,
            "neuromodulation_applied": neuromod_applied,
            "pack_applied": current_pack,
            "generation_time": generation_time
        }
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise

def generate_text(prompt: str, max_tokens: int = 100, 
                 temperature: float = 1.0, top_p: float = 1.0,
                 pack_name: str = None, custom_pack: Dict = None,
                 individual_effects: List[Dict] = None,
                 multiple_packs: List[str] = None) -> str:
    """Legacy generate_text method - use generate_text_with_probes for full features"""
    result = generate_text_with_probes(
        prompt, max_tokens, temperature, top_p,
        pack_name, custom_pack, individual_effects, multiple_packs
    )
    return result["text"]

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "vertex-ai-prediction-server",
        "timestamp": time.time(),
        "model_loaded": model is not None,
        "neuromodulation_available": NEUROMOD_AVAILABLE
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for Vertex AI with FULL PROBE SYSTEM"""
    try:
        # Parse request
        request_data = request.get_json()
        
        if not request_data or 'instances' not in request_data:
            return jsonify({"error": "Invalid request format"}), 400
        
        instances = request_data['instances']
        if not instances:
            return jsonify({"error": "No instances provided"}), 400
        
        # Process each instance
        predictions = []
        
        for instance in instances:
            # Extract parameters
            prompt = instance.get('prompt', '')
            max_tokens = instance.get('max_tokens', 100)
            temperature = instance.get('temperature', 1.0)
            top_p = instance.get('top_p', 1.0)
            
            # Extract neuromodulation parameters
            pack_name = instance.get('pack_name')
            custom_pack = instance.get('custom_pack')
            individual_effects = instance.get('individual_effects')
            multiple_packs = instance.get('multiple_packs')
            
            # Extract probe and emotion tracking parameters
            track_emotions = instance.get('emotion_tracking_enabled', True)
            neuromodulation_enabled = instance.get('neuromodulation_enabled', True)
            probe_system_enabled = instance.get('probe_system_enabled', True)
            
            if not prompt:
                predictions.append({"error": "No prompt provided"})
                continue
            
            try:
                # Generate text with FULL PROBE SYSTEM
                if neuromodulation_enabled and probe_system_enabled:
                    result = generate_text_with_probes(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        pack_name=pack_name,
                        custom_pack=custom_pack,
                        individual_effects=individual_effects,
                        multiple_packs=multiple_packs,
                        track_emotions=track_emotions
                    )
                    
                    # Extract components
                    generated_text = result["text"]
                    probe_data_result = result.get("probe_data", [])
                    probe_summary = result.get("probe_summary", {})
                    emotion_data = result.get("emotions", {})
                    
                    # Prepare comprehensive response
                    prediction_response = {
                        "generated_text": generated_text,
                        "probe_data": probe_data_result,
                        "probe_summary": probe_summary,
                        "emotions": emotion_data,
                        "neuromodulation_applied": result.get("neuromodulation_applied", False),
                        "pack_applied": result.get("pack_applied"),
                        "generation_time": result.get("generation_time", 0)
                    }
                    
                else:
                    # Fallback to basic generation
                    generated_text = generate_text(
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )
                    
                    prediction_response = {
                        "generated_text": generated_text,
                        "probe_data": [],
                        "probe_summary": {"total_probes": 0},
                        "emotions": {},
                        "neuromodulation_applied": False,
                        "pack_applied": None,
                        "generation_time": 0
                    }
                
                predictions.append(prediction_response)
                
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                predictions.append({"error": str(e)})
        
        # Return response
        response = {
            "predictions": predictions
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information with probe system details"""
    return jsonify({
        "model_name": model_name,
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "neuromodulation_available": NEUROMOD_AVAILABLE,
        "probe_hooks_active": len(probe_hooks),
        "emotion_tracking_active": emotion_tracker is not None,
        "current_pack": current_pack
    })

@app.route('/probe_status', methods=['GET'])
def probe_status():
    """Get probe system status and recent probe data"""
    return jsonify({
        "probe_system_active": len(probe_hooks) > 0,
        "total_probe_hooks": len(probe_hooks),
        "recent_probe_data": probe_data[-50:] if probe_data else [],  # Last 50 probes
        "probe_types_active": list(set(p['probe_type'] for p in probe_data)) if probe_data else [],
        "current_pack": current_pack,
        "timestamp": time.time()
    })

@app.route('/emotion_status', methods=['GET'])
def emotion_status():
    """Get emotion tracking system status"""
    if not emotion_tracker:
        return jsonify({"error": "Emotion tracking not available"}), 503
    
    try:
        # Get emotion summary for current session
        session_id = f"vertex_container_{int(time.time())}"
        summary = emotion_tracker.get_emotion_summary(session_id)
        
        return jsonify({
            "emotion_tracking_active": True,
            "emotion_summary": summary,
            "current_pack": current_pack,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error(f"Failed to get emotion status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/available_packs', methods=['GET'])
def available_packs():
    """Get list of available neuromodulation packs"""
    if not neuromod_tool or not NEUROMOD_AVAILABLE:
        return jsonify({"error": "Neuromodulation not available"}), 503
    
    try:
        # Get available packs from registry
        packs = neuromod_tool.registry.list_packs()
        return jsonify({
            "available_packs": packs,
            "total_count": len(packs) if isinstance(packs, list) else len(packs.keys())
        })
    except Exception as e:
        logger.error(f"Failed to get available packs: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/available_effects', methods=['GET'])
def available_effects():
    """Get list of available individual effects"""
    if not neuromod_tool or not NEUROMOD_AVAILABLE:
        return jsonify({"error": "Neuromodulation not available"}), 503
    
    try:
        # Get available effects from registry
        effects = neuromod_tool.pack_manager.effect_registry.list_effects()
        return jsonify({
            "available_effects": effects,
            "total_count": len(effects)
        })
    except Exception as e:
        logger.error(f"Failed to get available effects: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("🚀 Starting Vertex AI prediction server...")
        logger.info(f"🔧 Environment: MODEL_NAME={os.environ.get('MODEL_NAME', 'Not set')}")
        logger.info(f"🔧 Environment: NEUROMODULATION_ENABLED={os.environ.get('NEUROMODULATION_ENABLED', 'Not set')}")
        logger.info(f"🔧 Environment: PROBE_SYSTEM_ENABLED={os.environ.get('PROBE_SYSTEM_ENABLED', 'Not set')}")
        
        # Load model on startup
        if load_model():
            logger.info("🚀 Starting prediction server with FULL PROBE SYSTEM...")
            logger.info(f"🔌 Active probe hooks: {len(probe_hooks)}")
            logger.info(f"🎭 Emotion tracking: {'✅ Active' if emotion_tracker else '❌ Inactive'}")
            logger.info(f"🧠 Neuromodulation: {'✅ Active' if neuromod_tool else '❌ Inactive'}")
            logger.info("🌐 Starting Flask server on port 8080...")
            app.run(host='0.0.0.0', port=8080, debug=False)
        else:
            logger.error("Failed to load model, exiting...")
            exit(1)
    except Exception as e:
        logger.error(f"Critical error during startup: {e}")
        logger.error("Stack trace:", exc_info=True)
        exit(1)
