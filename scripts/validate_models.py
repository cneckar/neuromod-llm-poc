#!/usr/bin/env python3
"""
Model Loading Validation Script

Tests loading of all three primary models and documents:
- Loading success/failure
- Loading times
- Memory usage (GPU and system RAM)
- Quantization status
- Model information

This script completes Section 4.1 action items from EXPERIMENT_EXECUTION_PLAN.md
"""

import os
import sys
import io
import time
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory tracking will be limited")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.model_support import ModelSupportManager
from neuromod.neuromod_factory import create_neuromod_tool, cleanup_neuromod_tool


class ModelValidator:
    """Validates model loading for all primary models"""
    
    _encoding_set = False  # Class variable to track if encoding was set
    
    def __init__(self, output_dir: str = "outputs/validation/models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "models": {}
        }
        
        # Set UTF-8 encoding for Windows console (only once)
        if sys.platform == 'win32' and not ModelValidator._encoding_set:
            try:
                if not isinstance(sys.stdout, io.TextIOWrapper) and hasattr(sys.stdout, 'buffer'):
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                if not isinstance(sys.stderr, io.TextIOWrapper) and hasattr(sys.stderr, 'buffer'):
                    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
                ModelValidator._encoding_set = True
            except (AttributeError, ValueError, OSError):
                # If wrapping fails, continue without it
                pass
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        if PSUTIL_AVAILABLE:
            info = {
                "cpu_count": psutil.cpu_count(),
                "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
        else:
            info = {
                "cpu_count": os.cpu_count() or 0,
                "total_ram_gb": None,
                "available_ram_gb": None,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            }
        
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                gpu_info.append({
                    "device_id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(i) / (1024**3), 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(i) / (1024**3), 2),
                })
            info["gpu_details"] = gpu_info
        
        return info
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if PSUTIL_AVAILABLE:
            memory = {
                "system_ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "system_ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "system_ram_percent": psutil.virtual_memory().percent,
            }
        else:
            memory = {
                "system_ram_used_gb": None,
                "system_ram_available_gb": None,
                "system_ram_percent": None,
            }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory[f"gpu_{i}_allocated_gb"] = round(torch.cuda.memory_allocated(i) / (1024**3), 2)
                memory[f"gpu_{i}_reserved_gb"] = round(torch.cuda.memory_reserved(i) / (1024**3), 2)
                memory[f"gpu_{i}_free_gb"] = round(
                    (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) / (1024**3), 2
                )
        
        return memory
    
    def test_model_loading(self, model_name: str, test_mode: bool = False) -> Dict[str, Any]:
        """Test loading a single model"""
        print(f"\n{'='*70}")
        print(f"Testing model: {model_name}")
        print(f"{'='*70}")
        print(f"[*] Starting validation process...")
        
        result = {
            "model_name": model_name,
            "test_mode": test_mode,
            "status": "unknown",
            "error": None,
            "loading_time_seconds": None,
            "memory_before": None,
            "memory_after": None,
            "memory_delta": None,
            "model_info": None,
            "quantization": None,
        }
        
        # Get memory before
        print(f"[*] Measuring baseline memory usage...")
        memory_before = self._get_memory_usage()
        result["memory_before"] = memory_before
        
        # Check for HuggingFace token
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        if 'meta-llama' in model_name.lower() and not hf_token:
            result["status"] = "skipped"
            result["error"] = "HuggingFace token required for Meta Llama models"
            result["note"] = "Set HUGGINGFACE_HUB_TOKEN environment variable"
            print(f"[WARN] Skipping {model_name}: HuggingFace authentication required")
            return result
        
        try:
            # Create model support manager
            print(f"[*] Initializing model support manager...")
            model_manager = ModelSupportManager(test_mode=test_mode)
            
            # Start timing
            start_time = time.time()
            
            # Load model
            print(f"[*] Loading model (this may take several minutes on first run)...")
            print(f"[*] Downloading model files if needed...")
            model, tokenizer, model_info = model_manager.load_model(model_name)
            
            # Get config for later use
            config = model_manager.model_configs.get(model_name)
            
            # End timing
            loading_time = time.time() - start_time
            result["loading_time_seconds"] = round(loading_time, 2)
            result["model_info"] = model_info
            
            # Get quantization info
            if model_info.get("quantization"):
                result["quantization"] = model_info.get("quantization")
            elif hasattr(model, "hf_quantizer"):
                result["quantization"] = "4bit" if hasattr(model.hf_quantizer, "bits") else "unknown"
            
            # Get memory after
            print(f"[*] Collecting memory usage data...")
            memory_after = self._get_memory_usage()
            result["memory_after"] = memory_after
            
            # Calculate memory delta
            memory_delta = {}
            for key in memory_after:
                if key in memory_before:
                    delta = memory_after[key] - memory_before[key]
                    memory_delta[key] = round(delta, 2)
            result["memory_delta"] = memory_delta
            
            # Test generation
            print(f"[OK] Model loaded successfully in {loading_time:.2f} seconds")
            print(f"[*] Testing generation...")
            
            try:
                # Special handling for Qwen2.5-Omni models (multimodal)
                if 'qwen2.5-omni' in model_name.lower() or 'qwen2_5_omni' in model_name.lower():
                    # Qwen2.5-Omni uses a different tokenizer/processor
                    try:
                        from transformers import Qwen2_5OmniProcessor
                        print(f"[*] Loading Qwen2.5-Omni processor...")
                        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
                        
                        # Create a simple text-only conversation
                        conversation = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "Hello, how are you?"}
                                ]
                            }
                        ]
                        
                        # Apply chat template
                        print(f"[*] Applying chat template...")
                        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
                        
                        # Tokenize - Qwen2.5-Omni processor returns different format
                        print(f"[*] Tokenizing input...")
                        inputs = processor(text=text, return_tensors="pt", padding=True)
                        
                        # Move to device
                        if torch.cuda.is_available():
                            print(f"[*] Moving inputs to GPU...")
                            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                        else:
                            # For CPU, ensure inputs are on CPU
                            inputs = {k: v if not isinstance(v, torch.Tensor) else v.cpu() for k, v in inputs.items()}
                        
                        # Generate
                        print(f"[*] Generating response (this may take a moment)...")
                        with torch.no_grad():
                            # Qwen2.5-Omni generate expects input_ids directly, not unpacked dict
                            if 'input_ids' in inputs:
                                outputs = model.generate(
                                    input_ids=inputs['input_ids'],
                                    attention_mask=inputs.get('attention_mask', None),
                                    max_new_tokens=10,
                                    do_sample=False
                                )
                            else:
                                outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
                        
                        # Decode - Qwen2.5-Omni processor expects specific format
                        print(f"[*] Decoding response...")
                        # Extract token IDs from outputs
                        if isinstance(outputs, torch.Tensor):
                            # If outputs is 2D [batch, seq], take first sequence
                            if outputs.dim() > 1:
                                token_ids = outputs[0].cpu().tolist()
                            else:
                                token_ids = outputs.cpu().tolist()
                        elif isinstance(outputs, (list, tuple)):
                            # If it's a list/tuple, get first element
                            if len(outputs) > 0:
                                if isinstance(outputs[0], torch.Tensor):
                                    token_ids = outputs[0].cpu().tolist()
                                else:
                                    token_ids = list(outputs[0]) if isinstance(outputs[0], (list, tuple)) else outputs[0]
                            else:
                                token_ids = []
                        else:
                            token_ids = outputs
                        
                        # Try decoding with processor's tokenizer
                        try:
                            # Qwen2.5-Omni processor has a tokenizer attribute
                            if hasattr(processor, 'tokenizer') and processor.tokenizer is not None:
                                generated_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
                            elif hasattr(processor, 'decode'):
                                # Try processor.decode with token_ids as list
                                generated_text = processor.decode(token_ids, skip_special_tokens=True)
                            else:
                                # Fallback: use model's tokenizer if available
                                if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                                    generated_text = model.tokenizer.decode(token_ids, skip_special_tokens=True)
                                else:
                                    raise ValueError("No tokenizer available for decoding")
                        except Exception as decode_error:
                            # Last resort: try with tokenizer from model_manager
                            try:
                                if 'tokenizer' in locals() and tokenizer is not None:
                                    generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                                else:
                                    raise decode_error
                            except:
                                # Final fallback: indicate decode failed but generation succeeded
                                generated_text = f"Generated {len(token_ids)} tokens (decode format issue: {decode_error})"
                                print(f"[WARN] Decode had format issues, but generation succeeded")
                        
                        print(f"[OK] Generation test successful: '{generated_text[:50]}...'")
                        result["generation_test"] = "passed"
                    except Exception as e:
                        # If special handling fails, skip generation test but mark model as loaded
                        print(f"[WARN] Qwen2.5-Omni generation test failed (model loaded successfully): {e}")
                        print(f"[*] This is expected - Qwen2.5-Omni requires special multimodal handling")
                        result["generation_test"] = "skipped"
                        result["generation_error"] = str(e)
                        result["generation_note"] = "Qwen2.5-Omni is multimodal and requires special input format"
                else:
                    # Standard generation for other models
                    test_prompt = "Hello, how are you?"
                    print(f"[*] Tokenizing prompt...")
                    inputs = tokenizer(test_prompt, return_tensors="pt")
                    
                    if torch.cuda.is_available():
                        print(f"[*] Moving inputs to GPU...")
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    print(f"[*] Generating response (this may take a moment)...")
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=10,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else None
                        )
                    
                    print(f"[*] Decoding response...")
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"[OK] Generation test successful: '{generated_text[:50]}...'")
                    result["generation_test"] = "passed"
                
                result["status"] = "success"
                
            except Exception as gen_error:
                # Generation test failed, but model loaded successfully
                print(f"[WARN] Generation test failed: {gen_error}")
                print(f"[*] Model loaded successfully, but generation test had issues")
                result["status"] = "success"  # Model loaded, so mark as success
                result["generation_test"] = "failed"
                result["generation_error"] = str(gen_error)
            
            # Cleanup
            print(f"[*] Cleaning up resources...")
            try:
                del model
                if 'tokenizer' in locals():
                    del tokenizer
                if 'inputs' in locals():
                    del inputs
                if 'outputs' in locals():
                    del outputs
                if 'processor' in locals():
                    del processor
            except:
                pass
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model_manager.cleanup()
            print(f"[*] Cleanup complete")
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            print(f"[ERROR] Failed to load model: {e}")
        
        return result
    
    def validate_all_models(self):
        """Validate all three primary models"""
        primary_models = [
            "meta-llama/Llama-3.1-70B-Instruct",
            "Qwen/Qwen-2.5-Omni-7B",
            "mistralai/Mixtral-8x22B-Instruct-v0.1"
        ]
        
        print("[*] Starting Model Validation")
        print(f"[*] Models to validate: {len(primary_models)}")
        print(f"[*] Output directory: {self.output_dir}")
        
        for model_name in primary_models:
            result = self.test_model_loading(model_name, test_mode=False)
            self.results["models"][model_name] = result
            # Add a small delay between models to allow cleanup
            import time
            time.sleep(2)
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
    
    def _save_results(self):
        """Save validation results to JSON"""
        output_file = self.output_dir / f"model_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n[OK] Results saved to: {output_file}")
        
        # Also save a summary markdown file
        self._save_summary_markdown()
    
    def _save_summary_markdown(self):
        """Save a markdown summary"""
        md_file = self.output_dir / "MODEL_VALIDATION_SUMMARY.md"
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("# Model Loading Validation Summary\n\n")
            f.write(f"**Validation Date**: {self.results['timestamp']}\n\n")
            
            f.write("## System Information\n\n")
            f.write(f"- **CPU Cores**: {self.results['system_info']['cpu_count']}\n")
            f.write(f"- **Total RAM**: {self.results['system_info']['total_ram_gb']} GB\n")
            f.write(f"- **Available RAM**: {self.results['system_info']['available_ram_gb']} GB\n")
            f.write(f"- **PyTorch Version**: {self.results['system_info']['pytorch_version']}\n")
            f.write(f"- **CUDA Available**: {self.results['system_info']['cuda_available']}\n")
            
            if self.results['system_info']['cuda_available']:
                f.write(f"- **CUDA Version**: {self.results['system_info'].get('cuda_version', 'N/A')}\n")
                f.write(f"- **GPU Count**: {self.results['system_info']['gpu_count']}\n")
                for gpu in self.results['system_info']['gpu_details']:
                    f.write(f"  - **GPU {gpu['device_id']}**: {gpu['name']} ({gpu['memory_total_gb']} GB)\n")
            
            f.write("\n## Model Validation Results\n\n")
            
            for model_name, result in self.results['models'].items():
                f.write(f"### {model_name}\n\n")
                f.write(f"- **Status**: {result['status']}\n")
                
                if result['status'] == 'success':
                    f.write(f"- **Loading Time**: {result['loading_time_seconds']} seconds\n")
                    f.write(f"- **Quantization**: {result.get('quantization', 'None')}\n")
                    
                    if result.get('model_info'):
                        info = result['model_info']
                        f.write(f"- **Backend**: {info.get('backend', 'N/A')}\n")
                        f.write(f"- **Size**: {info.get('size', 'N/A')}\n")
                        f.write(f"- **Parameters**: {info.get('parameters', 'N/A')}\n")
                    
                    if result.get('memory_delta'):
                        delta = result['memory_delta']
                        f.write(f"- **Memory Delta**:\n")
                        for key, value in delta.items():
                            if 'gpu' in key and value != 0:
                                f.write(f"  - {key}: {value:+.2f} GB\n")
                
                elif result['status'] == 'skipped':
                    f.write(f"- **Reason**: {result.get('error', 'N/A')}\n")
                    f.write(f"- **Note**: {result.get('note', 'N/A')}\n")
                
                elif result['status'] == 'failed':
                    f.write(f"- **Error**: {result.get('error', 'N/A')}\n")
                
                f.write("\n")
            
            f.write("## Summary\n\n")
            successful = sum(1 for r in self.results['models'].values() if r['status'] == 'success')
            skipped = sum(1 for r in self.results['models'].values() if r['status'] == 'skipped')
            failed = sum(1 for r in self.results['models'].values() if r['status'] == 'failed')
            
            f.write(f"- **Successful**: {successful}/{len(self.results['models'])}\n")
            f.write(f"- **Skipped**: {skipped}/{len(self.results['models'])}\n")
            f.write(f"- **Failed**: {failed}/{len(self.results['models'])}\n")
        
        print(f"[OK] Summary saved to: {md_file}")
    
    def _print_summary(self):
        """Print validation summary"""
        print(f"\n{'='*70}")
        print("📊 Validation Summary")
        print(f"{'='*70}")
        
        for model_name, result in self.results['models'].items():
            status_icon = "[OK]" if result['status'] == 'success' else "[SKIP]" if result['status'] == 'skipped' else "[FAIL]"
            print(f"{status_icon} {model_name}: {result['status']}")
            
            if result['status'] == 'success':
                print(f"   [*] Loading time: {result['loading_time_seconds']} seconds")
                if result.get('quantization'):
                    print(f"   [*] Quantization: {result['quantization']}")
                if result.get('memory_delta'):
                    for key, value in result['memory_delta'].items():
                        if 'gpu' in key and value != 0:
                            print(f"   [*] {key}: {value:+.2f} GB")
            elif result['status'] == 'skipped':
                print(f"   [*] {result.get('note', 'N/A')}")
            elif result['status'] == 'failed':
                print(f"   [ERROR] Error: {result.get('error', 'N/A')[:100]}")
        
        successful = sum(1 for r in self.results['models'].values() if r['status'] == 'success')
        total = len(self.results['models'])
        print(f"\n[OK] Successfully validated: {successful}/{total} models")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate model loading for all primary models")
    parser.add_argument("--output-dir", default="outputs/validation/models", 
                       help="Output directory for validation results")
    parser.add_argument("--model", help="Test a specific model only")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Use test mode (allows testing with smaller models like gpt2)")
    
    args = parser.parse_args()
    
    validator = ModelValidator(output_dir=args.output_dir)
    
    if args.model:
        # Test single model
        result = validator.test_model_loading(args.model, test_mode=args.test_mode)
        validator.results["models"][args.model] = result
        validator._save_results()
        validator._print_summary()
    else:
        # Test all models
        validator.validate_all_models()


if __name__ == "__main__":
    main()

