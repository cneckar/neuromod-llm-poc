"""
Base Test Class for Neuromodulation Testing
"""

import torch
import os
import gc
import math
import statistics
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable MPS completely
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

class BaseTest(ABC):
    """Base class for all neuromodulation tests"""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        
    def load_model(self):
        """Load model with conservative settings"""
        print(f"Loading {self.model_name} model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        model = model.cpu()
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        print(f"✅ {self.model_name} model loaded successfully")
        return model, tokenizer

    def set_neuromod_tool(self, neuromod_tool):
        """Set the neuromodulation tool for this test"""
        self.neuromod_tool = neuromod_tool

    def generate_response_safe(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate response with very safe settings to avoid bus errors"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Get neuromodulation effects if available
            logits_processors = []
            gen_kwargs = {}
            
            if self.neuromod_tool:
                # Update token position for phase-based effects
                self.neuromod_tool.update_token_position(0)  # Reset for each new prompt
                logits_processors = self.neuromod_tool.get_logits_processors()
                gen_kwargs = self.neuromod_tool.get_generation_kwargs()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    early_stopping=False,
                    logits_processor=logits_processors,
                    **gen_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            return response if response else "0"
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "0"

    def extract_rating_improved(self, response: str) -> int:
        """Improved rating extraction that handles various response formats"""
        response = response.strip().lower()
        
        patterns = [
            r'\b([0-4])\b',                    # Single digit 0-4
            r'(\d+)/10',                       # X/10 format
            r'(\d+) out of',                   # X out of format
            r'(\d+) outof',                    # X outof format
            r'(\d+)\+',                        # X+ format
            r'(\d+)\.(\d+)',                   # X.Y decimal format
            r'(\d+)/5',                        # X/5 format
            r'(\d+) out\(',                    # X out( format
            r'(\d+) out',                      # X out format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                if pattern == r'(\d+)\.(\d+)':
                    whole = int(match.group(1))
                    decimal = int(match.group(2))
                    if whole == 0 and decimal > 0:
                        return min(4, decimal)
                    else:
                        return min(4, whole)
                else:
                    num = int(match.group(1))
                    if pattern == r'(\d+)/10':
                        return min(4, max(0, round(num / 2.5)))
                    elif pattern == r'(\d+)/5':
                        return min(4, max(0, round(num * 0.8)))
                    else:
                        return min(4, max(0, num))
        
        # Handle special cases
        if 'a+' in response or 'excellent' in response:
            return 4
        elif 'very good' in response:
            return 3
        elif 'good' in response:
            return 2
        elif 'fair' in response or 'okay' in response:
            return 1
        elif 'poor' in response or 'bad' in response:
            return 0
        elif 'n/a' in response or 'none' in response:
            return 0
        
        return 0

    @abstractmethod
    def run_test(self, neuromod_tool=None, **kwargs) -> Dict[str, Any]:
        """Run the test and return results"""
        pass

    @abstractmethod
    def get_test_name(self) -> str:
        """Return the name of this test"""
        pass

    def cleanup(self):
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
