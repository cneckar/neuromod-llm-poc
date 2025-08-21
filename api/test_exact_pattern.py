#!/usr/bin/env python3
"""
Test script using the exact same pattern as working neuromod tests
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Disable MPS completely for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_exact_pattern():
    """Test using the exact same pattern as working neuromod tests"""
    try:
        logger.info("Loading GPT-2 model with exact neuromod test pattern...")
        
        # Load tokenizer (exact same as neuromod tests)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model (exact same as neuromod tests)
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Force CPU and eval mode (exact same as neuromod tests)
        model = model.cpu()
        model.eval()
        
        logger.info("GPT-2 loaded successfully")
        
        # Test generation with exact same pattern as neuromod tests
        prompt = "Hello"
        max_tokens = 5
        
        logger.info(f"Generating text for prompt: '{prompt}' with max_tokens: {max_tokens}")
        
        # Tokenize (exact same as neuromod tests)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        logger.info(f"Input keys: {list(inputs.keys())}")
        logger.info(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        
        # Generate (exact same as neuromod tests)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=False
            )
        
        # Decode (exact same as neuromod tests)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        logger.info(f"Generated text: '{response}'")
        logger.info("✅ Exact pattern test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_exact_pattern()
    if success:
        print("\n🎉 Exact pattern test passed!")
    else:
        print("\n❌ Exact pattern test failed!")
