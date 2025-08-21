#!/usr/bin/env python3
"""
Test script using the fixed model loading and generation approach
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

def test_fixed_gpt2():
    """Test GPT-2 with the fixed approach"""
    try:
        logger.info("Loading GPT-2 model with conservative settings...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with conservative settings (same as working neuromod tests)
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Force CPU and eval mode
        model = model.cpu()
        model.eval()
        
        logger.info("GPT-2 loaded successfully")
        
        # Test generation with safe settings
        prompt = "Hello"
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Tokenize with proper attention mask
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True,
            padding=True,
            return_attention_mask=True
        )
        
        logger.info(f"Input shape: {inputs['input_ids'].shape}")
        logger.info(f"Attention mask shape: {inputs['attention_mask'].shape}")
        
        # Generate with safe settings (same as working neuromod tests)
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=5,
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
    success = test_fixed_gpt2()
    if success:
        print("\n🎉 Fixed GPT-2 generation test passed!")
    else:
        print("\n❌ Fixed GPT-2 generation test failed!")
