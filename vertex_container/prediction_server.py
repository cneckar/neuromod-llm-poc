"""
Vertex AI Custom Prediction Server
Handles model inference with neuromodulation effects
"""

import os
import json
import logging
import time
from typing import Dict, Any
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
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
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

def generate_text(prompt: str, max_tokens: int = 100, 
                 temperature: float = 1.0, top_p: float = 1.0,
                 pack_name: str = None) -> str:
    """Generate text with optional neuromodulation effects"""
    global model, tokenizer, neuromod_tool
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")
    
    try:
        # Apply neuromodulation pack if specified
        if pack_name and neuromod_tool and NEUROMOD_AVAILABLE:
            try:
                neuromod_tool.load_pack(pack_name)
                logger.info(f"Applied pack: {pack_name}")
                
                # Apply effects to the loaded model
                neuromod_tool.apply_to_model(model)
                logger.info(f"Applied neuromodulation effects from pack: {pack_name}")
            except Exception as e:
                logger.warning(f"Failed to apply neuromodulation effects: {e}")
        elif pack_name and not NEUROMOD_AVAILABLE:
            logger.warning(f"Pack '{pack_name}' requested but neuromodulation not available")
        
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
            pack_name = instance.get('pack_name')
            
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
                    pack_name=pack_name
                )
                
                predictions.append({
                    "generated_text": generated_text,
                    "pack_applied": pack_name
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

if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Starting prediction server...")
        app.run(host='0.0.0.0', port=8080, debug=False)
    else:
        logger.error("Failed to load model, exiting...")
        exit(1)
