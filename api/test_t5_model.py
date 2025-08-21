#!/usr/bin/env python3
"""
Test script for T5 model (encoder-decoder architecture)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Disable MPS completely for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_t5_model():
    """Test T5 model with conservative settings"""
    try:
        logger.info("Loading T5-small model...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("t5-small")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with conservative settings
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "t5-small",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Force CPU and eval mode
        model = model.cpu()
        model.eval()
        
        logger.info("T5-small loaded successfully")
        
        # Test generation
        prompt = "translate English to French: Hello world"
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.cpu() for k, v in inputs.items()}
        
        logger.info(f"Input keys: {list(inputs.keys())}")
        logger.info(f"Input shapes: {[(k, v.shape) for k, v in inputs.items()]}")
        
        # Generate with safe settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=2,
                early_stopping=False
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: '{response}'")
        logger.info("✅ T5 test successful!")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_t5_model()
    if success:
        print("\n🎉 T5 model test passed!")
    else:
        print("\n❌ T5 model test failed!")
