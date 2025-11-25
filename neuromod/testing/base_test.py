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
            
            # Determine the device where inputs should be created
            # For quantized models with device_map, we need to find the primary device
            input_device = None
            if model is not None:
                try:
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        # For device_map models, find the primary device
                        device_map = model.hf_device_map
                        for key in ['', 'model.embed_tokens', 'model', 0]:
                            if key in device_map:
                                dev = device_map[key]
                                if isinstance(dev, int):
                                    input_device = torch.device(f"cuda:{dev}")
                                elif isinstance(dev, torch.device):
                                    input_device = dev
                                elif isinstance(dev, str):
                                    input_device = torch.device(dev)
                                break
                        if input_device is None and device_map:
                            first_val = next(iter(device_map.values()))
                            if isinstance(first_val, int):
                                input_device = torch.device(f"cuda:{first_val}")
                            elif isinstance(first_val, torch.device):
                                input_device = first_val
                    else:
                        # For non-device_map models, get device from parameters
                        input_device = next(model.parameters()).device
                except (StopIteration, AttributeError):
                    if torch.cuda.is_available():
                        input_device = torch.device("cuda:0")
                    else:
                        input_device = torch.device("cpu")
            else:
                input_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            
            # Tokenize and immediately move to the correct device
            # This ensures position IDs and other internal tensors are created on the right device
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: (v.to(input_device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            # Move inputs to the same device as the model
            if model is not None:
                # Get device from model - try multiple methods for robustness
                model_device = None
                try:
                    # Method 1: Try to get device from model's first parameter
                    model_device = next(model.parameters()).device
                except (StopIteration, AttributeError):
                    try:
                        # Method 2: Try to get device from model's device attribute
                        if hasattr(model, 'device'):
                            model_device = model.device
                        # Method 3: Check if model has a device_map and get the main device
                        elif hasattr(model, 'hf_device_map'):
                            device_map = model.hf_device_map
                            # Find the main device (usually device 0 or the first GPU)
                            if device_map:
                                # Get the first device from device_map
                                first_key = next(iter(device_map.values()))
                                if isinstance(first_key, (int, torch.device)):
                                    model_device = torch.device(f"cuda:{first_key}") if isinstance(first_key, int) else first_key
                                elif isinstance(first_key, dict) and 'device' in first_key:
                                    model_device = torch.device(first_key['device'])
                    except (AttributeError, TypeError):
                        pass
                
                # Fallback: use CUDA if available, else CPU
                if model_device is None:
                    if torch.cuda.is_available():
                        model_device = torch.device("cuda:0")
                    else:
                        model_device = torch.device("cpu")
                
                # Move all input tensors to the model's device
                inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) 
                         for k, v in inputs.items()}
                
                # Debug: verify device placement
                if hasattr(self, '_debug_device') and self._debug_device:
                    print(f"[DEBUG] Model device: {model_device}, Input devices: {[v.device if isinstance(v, torch.Tensor) else 'N/A' for v in inputs.values()]}")
            else:
                # No model loaded, use CPU
                inputs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            # Get neuromodulation effects if available
            logits_processors = []
            gen_kwargs = {}
            
            if self.neuromod_tool:
                # Update token position for phase-based effects
                self.neuromod_tool.update_token_position(0)  # Reset for each new prompt
                raw_processors = self.neuromod_tool.get_logits_processors()
                gen_kwargs = self.neuromod_tool.get_generation_kwargs()
                
                # Wrap logits processors to ensure device consistency
                if raw_processors:
                    from transformers import LogitsProcessor
                    
                    class DeviceAwareLogitsProcessor(LogitsProcessor):
                        """Wrapper that ensures input_ids and scores are on the same device"""
                        def __init__(self, processor):
                            self.processor = processor
                        
                        def __call__(self, input_ids, scores):
                            # Ensure input_ids is on the same device as scores
                            if input_ids.device != scores.device:
                                input_ids = input_ids.to(scores.device)
                            # Call the original processor
                            return self.processor(input_ids, scores)
                    
                    logits_processors = [DeviceAwareLogitsProcessor(p) if p is not None else None 
                                       for p in raw_processors]
                    logits_processors = [p for p in logits_processors if p is not None]
                else:
                    logits_processors = []
            
            # Final device check right before generation - ensure everything is on the same device
            if model is not None:
                try:
                    # Get the actual device the model will use for generation
                    # For quantized models with device_map, we need to be more careful
                    model_device = None
                    try:
                        model_device = next(model.parameters()).device
                    except (StopIteration, AttributeError):
                        # Check device_map if available
                        if hasattr(model, 'hf_device_map') and model.hf_device_map:
                            # Get the primary device from device_map
                            device_map = model.hf_device_map
                            # Usually the main device is device 0 or cuda:0
                            if torch.cuda.is_available():
                                model_device = torch.device("cuda:0")
                            else:
                                model_device = torch.device("cpu")
                        else:
                            model_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                    
                    # Force all input tensors to the model's device
                    inputs = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) 
                             for k, v in inputs.items()}
                except Exception as e:
                    # Last resort: try to move to CUDA if available
                    if torch.cuda.is_available():
                        inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) 
                                 for k, v in inputs.items()}
            
            # Ensure model is in eval mode and on the correct device
            if hasattr(model, 'eval'):
                model.eval()
            
            # Use the same generation approach as the chat interface
            # For models with device_map, we need to be extra careful about device placement
            # HuggingFace's generate() should handle device_map automatically, but we need to ensure
            # inputs are on the correct device for the first layer
            try:
                # Get the device where the input embeddings are (usually the first device in device_map)
                gen_device = None
                if hasattr(model, 'hf_device_map') and model.hf_device_map:
                    # For device_map models, find the primary device (usually where embeddings are)
                    device_map = model.hf_device_map
                    # Check common keys that indicate the main device
                    for key in ['', 'model.embed_tokens', 'model', 0]:
                        if key in device_map:
                            dev = device_map[key]
                            if isinstance(dev, int):
                                gen_device = torch.device(f"cuda:{dev}")
                            elif isinstance(dev, torch.device):
                                gen_device = dev
                            elif isinstance(dev, str):
                                gen_device = torch.device(dev)
                            break
                    # Fallback: use first device found
                    if gen_device is None and device_map:
                        first_val = next(iter(device_map.values()))
                        if isinstance(first_val, int):
                            gen_device = torch.device(f"cuda:{first_val}")
                        elif isinstance(first_val, torch.device):
                            gen_device = first_val
                
                # If still no device, try to get from model parameters
                if gen_device is None:
                    try:
                        gen_device = next(model.parameters()).device
                    except (StopIteration, AttributeError):
                        if torch.cuda.is_available():
                            gen_device = torch.device("cuda:0")
                        else:
                            gen_device = torch.device("cpu")
                
                # Ensure all inputs are on this device
                inputs = {k: (v.to(gen_device) if isinstance(v, torch.Tensor) else v) 
                         for k, v in inputs.items()}
            except Exception as e:
                # Last resort: use CUDA if available
                if torch.cuda.is_available():
                    gen_device = torch.device("cuda:0")
                    inputs = {k: (v.to(gen_device) if isinstance(v, torch.Tensor) else v) 
                             for k, v in inputs.items()}
                else:
                    gen_device = torch.device("cpu")
                    inputs = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) 
                             for k, v in inputs.items()}
            
            with torch.no_grad():
                # For quantized models with device_map, HuggingFace may create internal tensors on CPU
                # We need to ensure all generation happens on the model's device
                # Set the device explicitly for generation
                try:
                    # Get the primary device from the model
                    if hasattr(model, 'hf_device_map') and model.hf_device_map:
                        # For device_map models, use the primary device
                        primary_device = None
                        for key, value in model.hf_device_map.items():
                            if isinstance(value, (int, torch.device)):
                                primary_device = torch.device(f"cuda:{value}") if isinstance(value, int) else value
                                break
                        if primary_device is None:
                            primary_device = next(model.parameters()).device
                    else:
                        primary_device = next(model.parameters()).device
                    
                    # Ensure inputs are on the primary device
                    inputs = {k: (v.to(primary_device) if isinstance(v, torch.Tensor) else v) 
                             for k, v in inputs.items()}
                    
                    # Force HuggingFace to use the correct device by setting it in the model's config
                    # This helps ensure position IDs and other internal tensors are created on the right device
                    original_device = getattr(model, 'device', None)
                    if not hasattr(model, 'device'):
                        # For device_map models, we can't set a single device, but we can ensure
                        # the generation kwargs use the right device
                        pass
                    
                except Exception as e:
                    # If device detection fails, just proceed with inputs as-is
                    logger.warning(f"Device detection failed, proceeding anyway: {e}")
                
                # Generate with explicit device handling
                # For quantized models with device_map, HuggingFace may create position IDs on CPU
                # We need to ensure the model knows which device to use for internal tensors
                # Try to set a device attribute on the model if it doesn't exist
                if hasattr(model, 'hf_device_map') and model.hf_device_map and not hasattr(model, 'device'):
                    # For device_map models, set a device attribute to help HuggingFace
                    try:
                        primary_device = None
                        for key, value in model.hf_device_map.items():
                            if isinstance(value, (int, torch.device)):
                                primary_device = torch.device(f"cuda:{value}") if isinstance(value, int) else value
                                break
                        if primary_device is None:
                            primary_device = next(model.parameters()).device
                        # Set device attribute to help HuggingFace create tensors on the right device
                        model.device = primary_device
                    except:
                        pass
                
                # Patch model's prepare_inputs_for_generation to ensure position IDs are on GPU
                # This is needed because HuggingFace creates position IDs on CPU by default for quantized models
                original_prepare = None
                if hasattr(model, 'prepare_inputs_for_generation'):
                    original_prepare = model.prepare_inputs_for_generation
                    
                    def patched_prepare_inputs_for_generation(input_ids, **kwargs):
                        """Patched version that ensures all tensors are on the correct device"""
                        # Get the device from input_ids
                        device = input_ids.device
                        
                        # Call original method
                        prepared = original_prepare(input_ids, **kwargs)
                        
                        # Ensure all tensors in prepared dict are on the correct device
                        if isinstance(prepared, dict):
                            for key, value in prepared.items():
                                if isinstance(value, torch.Tensor) and value.device != device:
                                    prepared[key] = value.to(device)
                        elif isinstance(prepared, torch.Tensor) and prepared.device != device:
                            prepared = prepared.to(device)
                        
                        return prepared
                    
                    # Temporarily patch the method
                    model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
                
                # Also patch torch.arange to default to the model's device for position IDs
                # This is a more aggressive fix that ensures any position IDs created are on GPU
                original_arange = torch.arange
                model_device = next(model.parameters()).device
                
                def patched_arange(*args, device=None, **kwargs):
                    """Patched torch.arange that defaults to model device"""
                    if device is None:
                        # If no device specified, use the model's device
                        device = model_device
                    return original_arange(*args, device=device, **kwargs)
                
                # Temporarily patch torch.arange
                torch.arange = patched_arange
                
                # Generate - HuggingFace should respect input tensor devices
                # If it still creates position IDs on CPU, we'll catch and handle the error
                try:
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
                        logits_processor=logits_processors if logits_processors else None,
                        **gen_kwargs
                    )
                finally:
                    # Restore original method and torch.arange
                    if original_prepare is not None:
                        model.prepare_inputs_for_generation = original_prepare
                    torch.arange = original_arange
                except RuntimeError as e:
                    if "Expected all tensors to be on the same device" in str(e):
                        # HuggingFace created internal tensors on CPU
                        # Try to work around by ensuring model has device attribute
                        if hasattr(model, 'hf_device_map') and model.hf_device_map:
                            # Force device attribute
                            try:
                                primary_device = next(model.parameters()).device
                                model.device = primary_device
                                # Retry generation
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
                                    logits_processor=logits_processors if logits_processors else None,
                                    **gen_kwargs
                                )
                            except Exception as e2:
                                # If retry fails, raise original error
                                raise e
                    else:
                        raise
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            
            # Automatically track emotion changes (disabled for debugging)
            # if response and response != "0":
            #     self.track_emotion_change(response, f"Generated response to: {prompt[:50]}...")
            
            return response if response else "0"
            
        except Exception as e:
            error_msg = str(e)
            print(f"Generation error: {error_msg}")
            
            # If it's a device mismatch, provide more debugging info
            if "device" in error_msg.lower() and ("cuda" in error_msg.lower() or "cpu" in error_msg.lower()):
                try:
                    if model is not None:
                        model_devices = set()
                        try:
                            for param in model.parameters():
                                model_devices.add(str(param.device))
                        except:
                            pass
                        
                        input_devices = set()
                        for k, v in inputs.items():
                            if isinstance(v, torch.Tensor):
                                input_devices.add(str(v.device))
                        
                        print(f"[DEBUG] Model parameter devices: {model_devices}")
                        print(f"[DEBUG] Input tensor devices: {input_devices}")
                        if hasattr(model, 'hf_device_map'):
                            print(f"[DEBUG] Model device_map: {model.hf_device_map}")
                except:
                    pass
            
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
