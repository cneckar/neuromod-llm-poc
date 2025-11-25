#!/usr/bin/env python3
"""
Simple Generation Test

This test bypasses the accelerate library and directly tests model generation
to ensure the bus error fix is working.
"""

import unittest
import os
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Disable MPS to avoid issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

class TestSimpleGeneration(unittest.TestCase):
    """Test that basic model generation works without bus errors"""
    
    def setUp(self):
        """Set up test environment"""
        self.model_name = "gpt2"
        self.device = "cpu"  # Force CPU to avoid MPS issues
        
    def test_direct_generation(self):
        """Test direct model generation without accelerate"""
        print(f"\n🧪 Testing direct generation with {self.model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model directly without accelerate
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                dtype=torch.float32,
                device_map=None  # Don't use device_map to avoid accelerate
            )
            
            # Move to CPU manually
            model = model.to(self.device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test generation
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # Basic assertions
            self.assertIsInstance(generated_text, str)
            self.assertGreater(len(generated_text), 0)
            print(f"   ✅ Generated: {generated_text}")
            
        except Exception as e:
            self.fail(f"Direct generation failed with error: {e}")
    
    def test_forward_pass_direct(self):
        """Test model forward pass directly"""
        print(f"\n🧪 Testing forward pass directly")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model directly
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                device_map=None
            )
            
            # Move to CPU manually
            model = model.to(self.device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test forward pass
            prompt = "Hello"
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Check outputs
            self.assertIsNotNone(outputs.logits)
            self.assertEqual(outputs.logits.shape[0], 1)  # batch size
            self.assertEqual(outputs.logits.shape[1], len(inputs['input_ids'][0]))  # sequence length
            self.assertEqual(outputs.logits.shape[2], model.config.vocab_size)  # vocab size
            
            print(f"   ✅ Forward pass successful: {outputs.logits.shape}")
            
        except Exception as e:
            self.fail(f"Forward pass failed with error: {e}")
    
    def test_bus_error_regression(self):
        """Test specifically for bus error regression"""
        print(f"\n🧪 Testing for bus error regression")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # This is the exact setup that was causing bus errors
            model_name = "gpt2"
            device = "cpu"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map=None
            )
            model = model.to(device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # This is the exact generation call that was failing
            prompt = "Hello, how are you?"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    max_new_tokens=5,
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
            # If we get here without a bus error, the fix is working
            self.assertIsInstance(generated_text, str)
            print(f"   ✅ No bus error - fix is working: {generated_text}")
            
        except Exception as e:
            if "bus error" in str(e).lower():
                self.fail(f"BUS ERROR REGRESSION DETECTED: {e}")
            else:
                self.fail(f"Other error occurred: {e}")
    
    def test_multiple_prompts(self):
        """Test multiple prompts to ensure stability"""
        print(f"\n🧪 Testing multiple prompts")
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            # Load model
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float32,
                device_map=None
            )
            model = model.to(self.device)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test multiple prompts
            prompts = [
                "The weather today is",
                "I love programming because",
                "The future of AI is",
            ]
            
            for i, prompt in enumerate(prompts):
                inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs['input_ids'],
                        max_new_tokens=8,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = response[len(prompt):].strip()
                
                self.assertIsInstance(generated_text, str)
                self.assertGreater(len(generated_text), 0)
                print(f"   ✅ Prompt {i+1}: '{prompt}' → '{generated_text}'")
                
        except Exception as e:
            self.fail(f"Multiple prompts failed with error: {e}")

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
