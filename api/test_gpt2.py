#!/usr/bin/env python3
"""
Minimal test script for GPT-2 generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpt2():
    """Test basic GPT-2 generation"""
    try:
        logger.info("Loading GPT-2 model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with minimal settings
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32
        )
        
        model.eval()
        logger.info("GPT-2 loaded successfully")
        
        # Test generation with minimal parameters
        prompt = "Hello"
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Simple tokenization
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        logger.info(f"Input shape: {inputs.shape}")
        
        # Generate with minimal parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 10,  # Just add 10 tokens
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
    success = test_gpt2()
    if success:
        print("\n🎉 GPT-2 generation test passed!")
    else:
        print("\n❌ GPT-2 generation test failed!")
