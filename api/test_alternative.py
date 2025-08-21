#!/usr/bin/env python3
"""
Alternative test script using different generation approach
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_alternative_generation():
    """Test with alternative generation approach"""
    try:
        model_name = "distilgpt2"
        
        logger.info(f"Loading {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        logger.info(f"{model_name} loaded successfully")
        
        # Test generation with alternative approach
        prompt = "Hello"
        logger.info(f"Generating text for prompt: '{prompt}'")
        
        # Tokenize
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        logger.info(f"Input shape: {inputs.shape}")
        
        # Try alternative generation method
        try:
            # Method 1: Use forward pass instead of generate
            logger.info("Trying forward pass method...")
            
            with torch.no_grad():
                # Get logits from forward pass
                outputs = model(inputs)
                logits = outputs.logits
                
                # Sample next token manually
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
                # Concatenate
                generated_tokens = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
                
                # Decode
                generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()
                
                logger.info(f"Generated text (forward pass): '{generated_text}'")
                logger.info("✅ Forward pass method successful!")
                return True
                
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            
            # Method 2: Try with explicit attention mask
            logger.info("Trying explicit attention mask method...")
            
            attention_mask = torch.ones_like(inputs)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=inputs.shape[1] + 1,  # Just add 1 token
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            logger.info(f"Generated text (attention mask): '{generated_text}'")
            logger.info("✅ Attention mask method successful!")
            return True
        
    except Exception as e:
        logger.error(f"All methods failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_alternative_generation()
    if success:
        print("\n🎉 Alternative generation test passed!")
    else:
        print("\n❌ Alternative generation test failed!")
