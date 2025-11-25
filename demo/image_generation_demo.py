#!/usr/bin/env python3
"""
Image Generation Demo with Neuromodulation

This demo integrates Stable Diffusion with the existing neuromodulation framework
to generate images using the existing LSD pack without modifying any core components.
"""

import torch
from diffusers import StableDiffusionPipeline
import logging
from typing import Dict, Any, Optional
import time
import os

# Import the existing neuromodulation system
from neuromod import NeuromodTool
from neuromod.pack_system import PackRegistry
from neuromod.visual_effects import apply_visual_effects_to_generation, combine_visual_prompts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageNeuromodInterface:
    """
    Interface for image generation with neuromodulation effects.
    
    This class adapts the existing neuromodulation framework to work with
    Stable Diffusion without modifying any of the core packs, effects, or probes.
    """
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initialize the image generation interface with neuromodulation"""
        self.model_name = model_name
        self.pipeline = None
        self.neuromod_tool = None
        self.device = "cpu"  # Default to CPU for compatibility
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU (GPU not available)")
        
        self._load_pipeline()
        self._setup_neuromodulation()
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline"""
        try:
            logger.info(f"Loading Stable Diffusion model: {self.model_name}")
            
            # Load pipeline with memory optimizations
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_name,
                dtype=torch.float32,  # Use float32 for CPU compatibility
                use_safetensors=True,
                safety_checker=None,  # Disable safety checker for research
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory optimizations
            if self.device == "cuda":
                self.pipeline.enable_attention_slicing()
                self.pipeline.enable_vae_slicing()
            
            logger.info("Stable Diffusion pipeline loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Stable Diffusion pipeline: {e}")
            raise
    
    def _setup_neuromodulation(self):
        """Setup the neuromodulation system"""
        try:
            # Load the existing pack registry
            registry = PackRegistry('packs/config.json')
            
            # Try to load image-focused packs if available
            try:
                import json
                from neuromod.pack_system import Pack
                
                with open('packs/image_focused_packs.json', 'r') as f:
                    image_packs = json.load(f)
                    # Convert to Pack objects and merge into registry
                    for pack_name, pack_data in image_packs['packs'].items():
                        try:
                            pack_obj = Pack.from_dict(pack_data)
                            registry.packs[pack_name] = pack_obj
                        except Exception as e:
                            logger.warning(f"Could not load pack {pack_name}: {e}")
                logger.info("Loaded image-focused packs")
            except FileNotFoundError:
                logger.info("Image-focused packs not found, using standard packs")
            
            # Create a dummy model for the neuromod tool (we don't actually use it for generation)
            # This is just to initialize the neuromodulation system
            dummy_model = torch.nn.Linear(10, 10)
            dummy_tokenizer = type('DummyTokenizer', (), {
                'eos_token_id': 0,
                'pad_token_id': 0
            })()
            
            # Initialize the neuromod tool with the existing system
            self.neuromod_tool = NeuromodTool(registry, dummy_model, dummy_tokenizer)
            
            logger.info("Neuromodulation system initialized")
            
        except Exception as e:
            logger.error(f"Failed to setup neuromodulation: {e}")
            raise
    
    def apply_neuromodulation_pack(self, pack_name: str, intensity: float = 0.5):
        """
        Apply a neuromodulation pack to influence image generation.
        
        This adapts the existing pack effects to work with image generation
        by modifying the pipeline's sampling parameters.
        """
        try:
            # Load the pack using the existing system
            result = self.neuromod_tool.apply(pack_name, intensity=intensity)
            
            if result and result.get('ok'):
                logger.info(f"Applied neuromodulation pack: {pack_name} (intensity: {intensity})")
                
                # Get the pack from the registry since it's not in the result
                pack = self.neuromod_tool.registry.get_pack(pack_name)
                if pack:
                    # Adapt pack effects to image generation parameters
                    self._adapt_pack_to_image_generation(pack, intensity)
                
                return True
            else:
                logger.warning(f"Failed to apply pack: {pack_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error applying neuromodulation pack: {e}")
            return False
    
    def _adapt_pack_to_image_generation(self, pack, intensity: float):
        """
        Adapt neuromodulation pack effects to image generation parameters.
        
        This uses the new visual effects system for image-focused packs,
        and falls back to the old mapping for text-based packs.
        """
        # Handle both Pack objects and dictionaries
        if hasattr(pack, 'effects'):
            effects = pack.effects
        else:
            effects = pack.get('effects', [])
        
        # Default generation parameters
        base_params = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'eta': 0.0,
            'strength': 1.0
        }
        
        # Check if this is an image-focused pack (has visual effects)
        has_visual_effects = any(
            (hasattr(effect, 'effect') and effect.effect in [
                'color_bias', 'style_transfer', 'composition_bias', 
                'visual_entropy', 'synesthetic_mapping', 'motion_blur'
            ]) or (isinstance(effect, dict) and effect.get('effect') in [
                'color_bias', 'style_transfer', 'composition_bias', 
                'visual_entropy', 'synesthetic_mapping', 'motion_blur'
            ])
            for effect in effects
        )
        
        if has_visual_effects:
            # Use the new visual effects system
            self.generation_params = apply_visual_effects_to_generation(effects, base_params)
            
            # Scale effects by intensity
            for key in ['guidance_scale', 'num_inference_steps', 'eta']:
                if key in self.generation_params:
                    if key == 'num_inference_steps':
                        # For steps, we want to scale the increase
                        base_val = base_params.get(key, 50)
                        current_val = self.generation_params[key]
                        increase = current_val - base_val
                        self.generation_params[key] = int(base_val + (increase * intensity))
                    else:
                        # For other params, scale the deviation from base
                        base_val = base_params.get(key, 0.0)
                        current_val = self.generation_params[key]
                        deviation = current_val - base_val
                        self.generation_params[key] = base_val + (deviation * intensity)
        else:
            # Fall back to old text-based mapping
            self.generation_params = base_params.copy()
            
            for effect in effects:
                # Handle both EffectConfig objects and dictionaries
                if hasattr(effect, 'effect'):
                    effect_type = effect.effect
                    weight = effect.weight * intensity
                else:
                    effect_type = effect.get('effect', '')
                    weight = effect.get('weight', 0.0) * intensity
                
                # Map text generation effects to image generation parameters
                if effect_type == 'temperature':
                    # Temperature affects guidance scale (higher temp = lower guidance)
                    self.generation_params['guidance_scale'] *= (1.0 - weight * 0.3)
                    
                elif effect_type == 'entropy':
                    # Entropy affects number of steps (higher entropy = more steps)
                    self.generation_params['num_inference_steps'] = int(
                        self.generation_params['num_inference_steps'] * (1.0 + weight * 0.5)
                    )
                    
                elif effect_type == 'attention':
                    # Attention effects can influence the overall strength
                    self.generation_params['strength'] *= (1.0 + weight * 0.2)
                    
                elif effect_type == 'steering':
                    # Steering affects eta (noise level)
                    self.generation_params['eta'] = min(1.0, weight * 0.5)
        
        # Clamp parameters to reasonable ranges
        self.generation_params['guidance_scale'] = max(1.0, min(20.0, self.generation_params['guidance_scale']))
        self.generation_params['num_inference_steps'] = max(10, min(100, self.generation_params['num_inference_steps']))
        self.generation_params['strength'] = max(0.1, min(2.0, self.generation_params['strength']))
        
        logger.info(f"Adapted generation parameters: {self.generation_params}")
    
    def generate_image(self, prompt: str, pack_name: Optional[str] = None, 
                      intensity: float = 0.5, **kwargs) -> Dict[str, Any]:
        """
        Generate an image with optional neuromodulation effects.
        
        Args:
            prompt: Text prompt for image generation
            pack_name: Name of neuromodulation pack to apply
            intensity: Intensity of the neuromodulation effects
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary containing the generated image and metadata
        """
        try:
            start_time = time.time()
            
            # Apply neuromodulation if specified
            if pack_name:
                success = self.apply_neuromodulation_pack(pack_name, intensity)
                if not success:
                    logger.warning(f"Failed to apply pack {pack_name}, using default parameters")
            
            # Get generation parameters (either from neuromodulation or defaults)
            gen_params = getattr(self, 'generation_params', {
                'guidance_scale': 7.5,
                'num_inference_steps': 50,
                'eta': 0.0
            })
            
            # Override with any provided kwargs
            gen_params.update(kwargs)
            
            logger.info(f"Generating image with prompt: '{prompt}'")
            logger.info(f"Generation parameters: {gen_params}")
            
            # Don't add visual prompts - the effects should only modify generation behavior
            # The model should generate the normal prompt, but with modified internal processing
            enhanced_prompt = prompt
            
            # Generate the image
            with torch.no_grad():
                result = self.pipeline(
                    prompt=enhanced_prompt,
                    guidance_scale=gen_params['guidance_scale'],
                    num_inference_steps=gen_params['num_inference_steps'],
                    eta=gen_params['eta'],
                    **{k: v for k, v in gen_params.items() if k not in ['guidance_scale', 'num_inference_steps', 'eta', 'color_prompts', 'style_prompts', 'composition_prompts', 'detail_prompts', 'synesthetic_prompts', 'motion_prompts']}
                )
            
            generation_time = time.time() - start_time
            
            # Extract the image
            image = result.images[0] if result.images else None
            
            if image is None:
                raise RuntimeError("No image generated")
            
            # Prepare response
            response = {
                'image': image,
                'prompt': prompt,
                'pack_name': pack_name,
                'intensity': intensity,
                'generation_params': gen_params,
                'generation_time': generation_time,
                'success': True
            }
            
            logger.info(f"Image generated successfully in {generation_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error generating image: {e}")
            return {
                'image': None,
                'prompt': prompt,
                'pack_name': pack_name,
                'intensity': intensity,
                'error': str(e),
                'success': False
            }
    
    def clear_neuromodulation(self):
        """Clear any applied neuromodulation effects"""
        if self.neuromod_tool:
            self.neuromod_tool.clear()
        
        # Reset generation parameters to defaults
        self.generation_params = {
            'guidance_scale': 7.5,
            'num_inference_steps': 50,
            'eta': 0.0,
            'strength': 1.0
        }
        
        logger.info("Neuromodulation effects cleared")


def demo_lsd_image_generation():
    """Demo function to generate images with LSD neuromodulation"""
    
    print("🎨 LSD Image Generation Demo")
    print("=" * 50)
    
    # Initialize the image generation interface
    try:
        image_gen = ImageNeuromodInterface()
        print("✅ Image generation interface initialized")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test prompts for normal, everyday subjects (to see LSD effects)
    test_prompts = [
        "A peaceful garden with flowers and trees",
        "A cozy living room with furniture and books",
        "A quiet street in a suburban neighborhood", 
        "A simple bowl of fruit on a wooden table"
    ]
    
    # Generate images with and without LSD pack
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {prompt[:50]}... ---")
        
        # Generate without neuromodulation (baseline)
        print("🔄 Generating baseline image...")
        baseline_result = image_gen.generate_image(prompt)
        
        if baseline_result['success']:
            baseline_image = baseline_result['image']
            baseline_filename = f"outputs/reports/test_suite/baseline_test_{i}.png"
            baseline_image.save(baseline_filename)
            print(f"✅ Baseline image saved: {baseline_filename}")
        else:
            print(f"❌ Baseline generation failed: {baseline_result.get('error', 'Unknown error')}")
            continue
        
        # Generate with LSD pack
        print("🔄 Generating with LSD neuromodulation...")
        lsd_result = image_gen.generate_image(
            prompt=prompt,
            pack_name="lsd",
            intensity=0.7
        )
        
        if lsd_result['success']:
            lsd_image = lsd_result['image']
            lsd_filename = f"outputs/reports/test_suite/lsd_test_{i}.png"
            lsd_image.save(lsd_filename)
            print(f"✅ LSD image saved: {lsd_filename}")
            
            # Show generation parameters
            params = lsd_result['generation_params']
            print(f"   Generation time: {lsd_result['generation_time']:.2f}s")
            print(f"   Guidance scale: {params['guidance_scale']:.2f}")
            print(f"   Inference steps: {params['num_inference_steps']}")
        else:
            print(f"❌ LSD generation failed: {lsd_result.get('error', 'Unknown error')}")
        
        print("-" * 50)
    
    print("\n🎉 Demo completed! Check the generated images to see the LSD effects.")


def interactive_image_generation():
    """Interactive image generation with neuromodulation"""
    
    print("🎨 Interactive Image Generation with Neuromodulation")
    print("=" * 60)
    
    # Initialize the interface
    try:
        image_gen = ImageNeuromodInterface()
        print("✅ Ready for image generation")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print("\nAvailable commands:")
    print("  /lsd <prompt> - Generate with LSD pack (e.g., /lsd a cat sitting on a chair)")
    print("  /caffeine <prompt> - Generate with caffeine pack")
    print("  /clear - Clear neuromodulation effects")
    print("  /quit - Exit")
    print()
    print("💡 Try normal, everyday prompts to see how neuromodulation affects perception!")
    print("   Examples: 'a cat sitting on a chair', 'a cup of coffee', 'a tree in a park'")
    print()
    
    while True:
        try:
            user_input = input("🎨 Enter command or prompt: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                break
            elif user_input.lower() == '/clear':
                image_gen.clear_neuromodulation()
                print("✅ Neuromodulation effects cleared")
                continue
            elif user_input.startswith('/lsd '):
                prompt = user_input[5:].strip()
                if prompt:
                    print(f"🔄 Generating with LSD pack: '{prompt}'")
                    result = image_gen.generate_image(prompt, pack_name="lsd", intensity=0.7)
                    if result['success']:
                        filename = f"outputs/reports/test_suite/lsd_{int(time.time())}.png"
                        result['image'].save(filename)
                        print(f"✅ Image saved: {filename}")
                    else:
                        print(f"❌ Generation failed: {result.get('error')}")
                continue
            elif user_input.startswith('/caffeine '):
                prompt = user_input[10:].strip()
                if prompt:
                    print(f"🔄 Generating with caffeine pack: '{prompt}'")
                    result = image_gen.generate_image(prompt, pack_name="caffeine", intensity=0.6)
                    if result['success']:
                        filename = f"caffeine_{int(time.time())}.png"
                        result['image'].save(filename)
                        print(f"✅ Image saved: {filename}")
                    else:
                        print(f"❌ Generation failed: {result.get('error')}")
                continue
            else:
                # Default generation without neuromodulation
                print(f"🔄 Generating: '{user_input}'")
                result = image_gen.generate_image(user_input)
                if result['success']:
                    filename = f"outputs/reports/test_suite/baseline_{int(time.time())}.png"
                    result['image'].save(filename)
                    print(f"✅ Image saved: {filename}")
                else:
                    print(f"❌ Generation failed: {result.get('error')}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_lsd_image_generation()
    else:
        interactive_image_generation()
