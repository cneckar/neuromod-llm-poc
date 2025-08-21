#!/usr/bin/env python3
"""
Test script for a simple, lightweight model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_model():
    """Test with a simple, lightweight model"""
    try:
        # Try a smaller model first
        model_name = "distilgpt2"  # Much smaller than GPT-2
        
        logger.info(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        model.eval()
        logger.info(f"{model_name} loaded successfully")
        
        # Test generation
        prompt = "Hello"
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Simple tokenization
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        logger.info(f"Input shape: {inputs.shape}")
        
        # Generate with minimal parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 5,  # Just add 5 tokens
                do_sample=False,  # Deterministic generation
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        logger.info(f"Generated text: '{generated_text}'")
        logger.info("✅ Test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_model()
    if success:
        print("\n🎉 Simple model generation test passed!")
    else:
        print("\n❌ Simple model generation test failed!")
