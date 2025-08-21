"""
Vertex AI Custom Prediction Server
Handles model inference with neuromodulation effects
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

# Import neuromodulation system
try:
    from neuromod import NeuromodTool
    from neuromod.effects import EffectRegistry
    from neuromod.pack_system import Pack, EffectConfig
    NEUROMOD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Neuromodulation system not available: {e}")
    NEUROMOD_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
neuromod_tool = None
model_name = None

def load_model():
    """Load the model and tokenizer"""
    global model, tokenizer, neuromod_tool, model_name
    
    model_name = os.environ.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B")
    
    logger.info(f"Loading model: {model_name}")
    
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
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                token=hf_token
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
        
        # Initialize neuromodulation tool
        if NEUROMOD_AVAILABLE:
            neuromod_tool = NeuromodTool()
            logger.info("Neuromodulation tool initialized")
        else:
            neuromod_tool = None
            logger.warning("Neuromodulation tool not available")
        
        logger.info("Model loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

def apply_neuromodulation(pack_name: str = None, custom_pack: Dict = None, 
                         individual_effects: List[Dict] = None, 
                         multiple_packs: List[str] = None) -> bool:
    """Apply neuromodulation effects in various ways"""
    global neuromod_tool
    
    if not neuromod_tool or not NEUROMOD_AVAILABLE:
        logger.warning("Neuromodulation not available")
        return False
    
    try:
        # Clear any existing effects first
        neuromod_tool.clear()
        
        # Method 1: Single predefined pack
        if pack_name:
            neuromod_tool.load_pack(pack_name)
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
            neuromod_tool.pack_manager.apply_pack(pack, model)
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
                neuromod_tool.pack_manager.apply_pack(single_pack, model)
                logger.info(f"Applied individual effect: {effect_name}")
        
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
            logger.info(f"Applied multiple packs: {multiple_packs}")
        
        # Apply effects to the model
        if any([pack_name, custom_pack, individual_effects, multiple_packs]):
            neuromod_tool.apply_to_model(model)
            logger.info("Applied neuromodulation effects to model")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to apply neuromodulation: {e}")
        return False

def generate_text(prompt: str, max_tokens: int = 100, 
                 temperature: float = 1.0, top_p: float = 1.0,
                 pack_name: str = None, custom_pack: Dict = None,
                 individual_effects: List[Dict] = None,
                 multiple_packs: List[str] = None) -> str:
    """Generate text with optional neuromodulation effects"""
    global model, tokenizer, neuromod_tool
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    try:
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
        
        # Generate with neuromodulation effects applied
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
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_name,
        "timestamp": time.time()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for Vertex AI"""
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
            
            if not prompt:
                predictions.append({"error": "No prompt provided"})
                continue
            
            try:
                # Generate text
                generated_text = generate_text(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pack_name=pack_name,
                    custom_pack=custom_pack,
                    individual_effects=individual_effects,
                    multiple_packs=multiple_packs
                )
                
                # Determine what was applied
                neuromod_info = {}
                if pack_name:
                    neuromod_info["pack_applied"] = pack_name
                elif custom_pack:
                    neuromod_info["custom_pack_applied"] = custom_pack.get('name', 'custom')
                elif individual_effects:
                    neuromod_info["individual_effects_applied"] = [e.get('effect') for e in individual_effects]
                elif multiple_packs:
                    neuromod_info["multiple_packs_applied"] = multiple_packs
                
                predictions.append({
                    "generated_text": generated_text,
                    **neuromod_info
                })
                
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
    """Get model information"""
    return jsonify({
        "model_name": model_name,
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

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
    # Load model on startup
    if load_model():
        logger.info("Starting prediction server...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model, exiting...")
        exit(1)
