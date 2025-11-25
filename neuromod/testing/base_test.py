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
# Import centralized model support
from ..model_support import create_model_support
from ..neuromod_factory import create_neuromod_tool, cleanup_neuromod_tool

# Import the simple emotion tracker
from .simple_emotion_tracker import SimpleEmotionTracker

# Disable MPS completely to avoid bus errors
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

class BaseTest(ABC):
    """Base class for all neuromodulation tests"""
    
    def __init__(self, model_name: str = None, test_mode: bool = True):
        self.model_name = model_name
        self.test_mode = test_mode
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        self.model_manager = None
        
        # Initialize simple emotion tracker
        self.emotion_tracker = SimpleEmotionTracker()
        self.current_test_id = None
        self.previous_response = None
        
    def start_emotion_tracking(self, test_id: str):
        """Start emotion tracking for a specific test"""
        self.current_test_id = test_id
        self.previous_response = None
        print(f"🎭 Starting emotion tracking for test: {test_id}")
        
    def track_emotion_change(self, response: str, context: str = ""):
        """Track emotional changes in the response"""
        if not self.current_test_id:
            return
            
        # Assess emotion change
        state = self.emotion_tracker.assess_emotion_change(
            response, 
            self.current_test_id, 
            self.previous_response
        )
        
        # Show emotion changes
        changes = []
        for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']:
            change = getattr(state, emotion)
            if change != "stable":
                changes.append(f"{emotion}: {change}")
        
        if changes:
            print(f"🎭 Emotions: {', '.join(changes)} | Valence: {state.valence}")
        
        # Update for next comparison
        self.previous_response = response
        
    def get_emotion_summary(self) -> Dict[str, any]:
        """Get emotion summary for the current test"""
        if not self.current_test_id:
            return {"error": "No active test"}
        return self.emotion_tracker.get_emotion_summary(self.current_test_id)
        
    def export_emotion_results(self, filename: str = None):
        """Export emotion tracking results"""
        if not filename:
            filename = f"outputs/reports/emotion/emotion_results_{self.current_test_id}.json"
        
        # Create directory if it doesn't exist
        from pathlib import Path
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.emotion_tracker.export_results(str(output_path))
            print(f"💾 Emotion results exported to: {filename}")
        except Exception as e:
            # Don't fail the test if emotion export fails
            print(f"⚠️  Could not export emotion results: {e}")
    
    def load_model(self):
        """Load model using centralized model support system"""
        try:
            # Create model support manager
            self.model_manager = create_model_support(test_mode=self.test_mode)
            
            # Get model name if not specified
            if self.model_name is None:
                self.model_name = self.model_manager.get_recommended_model()
            
            print(f"Loading {self.model_name} model...")
            
            # Load model using centralized system
            self.model, self.tokenizer, model_info = self.model_manager.load_model(
                self.model_name
            )
            
            print(f"✅ Model loaded successfully")
            print(f"   Backend: {model_info.get('backend', 'unknown')}")
            print(f"   Size: {model_info.get('size', 'unknown')}")
            print(f"   Parameters: {model_info.get('parameters', 'unknown')}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def set_neuromod_tool(self, neuromod_tool):
        """Set the neuromodulation tool for this test"""
        self.neuromod_tool = neuromod_tool

    def generate_response_safe(self, prompt: str, max_tokens: int = 5) -> str:
        """Generate response with very safe settings and probe monitoring"""
        try:
            # Get tokenizer from neuromod tool if available
            tokenizer = self.tokenizer
            model = self.model
            
            if self.neuromod_tool:
                tokenizer = self.neuromod_tool.tokenizer
                model = self.neuromod_tool.model
            
            if tokenizer is None:
                raise ValueError("No tokenizer available")
                
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            
            # Move inputs to the same device as the model
            if model is not None:
                # Get device from model's first parameter
                try:
                    model_device = next(model.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except (StopIteration, AttributeError):
                    # Fallback: check if CUDA is available
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    else:
                        inputs = {k: v.cpu() for k, v in inputs.items()}
            else:
                # No model loaded, use CPU
                inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Get neuromodulation effects if available
            logits_processors = []
            gen_kwargs = {}
            
            if self.neuromod_tool:
                # Update token position for phase-based effects
                self.neuromod_tool.update_token_position(0)  # Reset for each new prompt
                logits_processors = self.neuromod_tool.get_logits_processors()
                gen_kwargs = self.neuromod_tool.get_generation_kwargs()
            
            # Use the same generation approach as the chat interface
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
                    early_stopping=False,
                    logits_processor=logits_processors,
                    **gen_kwargs
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Automatically track emotion changes (disabled for debugging)
            # if response and response != "0":
            #     self.track_emotion_change(response, f"Generated response to: {prompt[:50]}...")
            
            return response if response else "0"
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "0"
    
    def _generate_with_probe_monitoring(self, inputs, max_tokens, logits_processors, gen_kwargs):
        """Generate response with real-time probe monitoring"""
        try:
            # Ensure inputs are on the correct device
            if self.neuromod_tool and self.neuromod_tool.model is not None:
                try:
                    model_device = next(self.neuromod_tool.model.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except (StopIteration, AttributeError):
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    else:
                        inputs = {k: v.cpu() for k, v in inputs.items()}
            
            # Use standard generation but with probe hooks
            with torch.no_grad():
                outputs = self.neuromod_tool.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.neuromod_tool.tokenizer.eos_token_id,
                    eos_token_id=self.neuromod_tool.tokenizer.eos_token_id,
                    use_cache=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=2,
                    early_stopping=False,
                    logits_processor=logits_processors,
                    **gen_kwargs
                )
            
            # Extract the generated part
            response = self.neuromod_tool.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(self.neuromod_tool.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
            
            # Update token position for probes after generation
            if self.neuromod_tool:
                self.neuromod_tool.update_token_position(max_tokens)
            
            # Track emotion changes
            if response and response != "0":
                self.track_emotion_change(response, "Probe-monitored generation")
            
            return response if response else "0"
            
        except Exception as e:
            print(f"Probe monitoring generation error: {e}")
            # Fallback to simple generation
            return self._generate_simple_fallback(inputs, max_tokens)
    
    def _generate_simple_fallback(self, inputs, max_tokens):
        """Simple fallback generation method"""
        try:
            # Ensure inputs are on the correct device
            if self.model is not None:
                try:
                    model_device = next(self.model.parameters()).device
                    inputs = {k: v.to(model_device) for k, v in inputs.items()}
                except (StopIteration, AttributeError):
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    else:
                        inputs = {k: v.cpu() for k, v in inputs.items()}
            
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
                    early_stopping=False
                )
            
            # Use appropriate tokenizer (neuromod_tool if available, else self.tokenizer)
            tokenizer = self.neuromod_tool.tokenizer if self.neuromod_tool else self.tokenizer
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
            return response if response else "0"
            
        except Exception as e:
            print(f"Fallback generation error: {e}")
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
        """Clean up resources using centralized system"""
        if self.model_manager:
            self.model_manager.cleanup()
        
        # Clear references
        self.model = None
        self.tokenizer = None
        self.neuromod_tool = None
        self.model_manager = None
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"🧹 Cleaned up {self.model_name} model")
