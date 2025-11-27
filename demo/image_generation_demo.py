#!/usr/bin/env python3
"""
Image Generation Demo with Neuromodulation

This demo integrates Stable Diffusion with the existing neuromodulation framework
to generate images using any available pack without modifying any core components.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, Optional, List
import time
import os
import argparse
import json

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import diffusers pipelines
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
# Try to import Turbo pipeline, but fall back to SDXL if not available
try:
    from diffusers import StableDiffusionXLTurboPipeline
    TURBO_PIPELINE_AVAILABLE = True
except ImportError:
    TURBO_PIPELINE_AVAILABLE = False
    logger.info("StableDiffusionXLTurboPipeline not available, will use StableDiffusionXLPipeline for Turbo models")

# Import the existing neuromodulation system
from neuromod import NeuromodTool
from neuromod.pack_system import PackRegistry
from neuromod.visual_effects import apply_visual_effects_to_generation, combine_visual_prompts

# Common Stable Diffusion models
COMMON_MODELS = {
    "sd-v1-5": "runwayml/stable-diffusion-v1-5",
    "sd-v1-4": "CompVis/stable-diffusion-v1-4",
    "sd-v2-1": "stabilityai/stable-diffusion-2-1",
    "sd-xl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}


class FrequencyAnalyzer:
    """Helper to analyze and visualize images/latents in the frequency domain."""
    
    @staticmethod
    def compute_fft_magnitude(data: np.ndarray) -> np.ndarray:
        """Compute the log-magnitude spectrum of a 2D image/channel."""
        # Compute 2D FFT
        f = np.fft.fft2(data)
        # Shift zero frequency to center
        fshift = np.fft.fftshift(f)
        # Compute magnitude spectrum (log scale for visibility)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        return magnitude_spectrum

    @staticmethod
    def save_spectral_analysis(image, latents, title, filename):
        """
        Generate and save a side-by-side spectral analysis of Pixel vs Latent space.
        
        Args:
            image: PIL Image (Pixel space)
            latents: torch.Tensor (Latent space, 1x4x64x64 or similar)
            title: Title for the plot
            filename: Output filename
        """
        # 1. Pixel Space Analysis
        # Convert to grayscale for simple frequency analysis
        img_gray = np.array(image.convert('L'))
        pixel_fft = FrequencyAnalyzer.compute_fft_magnitude(img_gray)

        # 2. Latent Space Analysis (Pre-Convolution)
        # Latents are typically 1x4x64x64. We analyze the 4 channels.
        latents_np = latents.detach().cpu().numpy()
        if len(latents_np.shape) == 4:
            latents_np = latents_np[0]  # Shape (4, 64, 64) or (4, H, W)
        elif len(latents_np.shape) == 3:
            # Already in (4, H, W) format
            pass
        else:
            logger.warning(f"Unexpected latent shape: {latents_np.shape}, skipping latent analysis")
            latents_np = np.zeros((4, 64, 64))
        
        # Ensure we have 4 channels
        if latents_np.shape[0] > 4:
            latents_np = latents_np[:4]
        elif latents_np.shape[0] < 4:
            # Pad with zeros if needed
            padding = np.zeros((4 - latents_np.shape[0], *latents_np.shape[1:]))
            latents_np = np.concatenate([latents_np, padding], axis=0)
        
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Spectral Analysis: {title}", fontsize=16)

        # --- Row 1: Visuals ---
        # Final Image
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Final Image (Pixel Space)")
        axs[0, 0].axis('off')

        # Latent Channels (Spatial)
        for i in range(4):
            axs[0, i+1].imshow(latents_np[i], cmap='viridis')
            axs[0, i+1].set_title(f"Latent Channel {i+1}\n(Spatial)")
            axs[0, i+1].axis('off')

        # --- Row 2: Frequency Domain ---
        # Pixel FFT
        axs[1, 0].imshow(pixel_fft, cmap='magma')
        axs[1, 0].set_title("Pixel FFT (Magnitude)")
        axs[1, 0].axis('off')

        # Latent FFTs
        for i in range(4):
            latent_fft = FrequencyAnalyzer.compute_fft_magnitude(latents_np[i])
            axs[1, i+1].imshow(latent_fft, cmap='magma')
            axs[1, i+1].set_title(f"Latent Ch {i+1} FFT\n(Frequency)")
            axs[1, i+1].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved spectral analysis to {filename}")

class ImageNeuromodInterface:
    """
    Interface for image generation with neuromodulation effects.
    
    This class adapts the existing neuromodulation framework to work with
    Stable Diffusion without modifying any of the core packs, effects, or probes.
    """
    
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5"):
        """Initialize the image generation interface with neuromodulation"""
        # Resolve model name if it's a shortcut
        if model_name in COMMON_MODELS:
            self.model_name = COMMON_MODELS[model_name]
            logger.info(f"Using model shortcut '{model_name}' -> '{self.model_name}'")
        else:
            self.model_name = model_name
        
        self.pipeline = None
        self.neuromod_tool = None
        self.registry = None
        self.device = "cpu"  # Default to CPU for compatibility
        
        # Check for GPU availability
        if torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU (GPU not available)")
        
        self._load_pipeline()
        self._setup_neuromodulation()
    
    def get_available_packs(self) -> List[str]:
        """Get list of all available packs"""
        if self.registry:
            return list(self.registry.packs.keys())
        return []
    
    def list_packs_by_category(self) -> Dict[str, List[str]]:
        """List packs organized by category"""
        if not self.registry:
            return {}
        
        # Try to get categories from metadata
        try:
            with open('packs/config.json', 'r') as f:
                config = json.load(f)
                if 'metadata' in config and 'categories' in config['metadata']:
                    categories = config['metadata']['categories']
                    # Filter to only include packs that actually exist
                    result = {}
                    for category, pack_list in categories.items():
                        existing_packs = [p for p in pack_list if p in self.registry.packs]
                        if existing_packs:
                            result[category] = existing_packs
                    return result
        except Exception as e:
            logger.debug(f"Could not load pack categories: {e}")
        
        # Fallback: return all packs in a single category
        return {"all": self.get_available_packs()}
    
    def _load_pipeline(self):
        """Load the Stable Diffusion pipeline"""
        try:
            logger.info(f"Loading Stable Diffusion model: {self.model_name}")
            
            # Determine which pipeline class to use based on model name
            if "sdxl-turbo" in self.model_name.lower() or "xl-turbo" in self.model_name.lower():
                # SDXL Turbo - use Turbo pipeline if available, otherwise use SDXL pipeline
                if TURBO_PIPELINE_AVAILABLE:
                    pipeline_class = StableDiffusionXLTurboPipeline
                    logger.info("Using StableDiffusionXLTurboPipeline for SDXL Turbo model")
                else:
                    pipeline_class = StableDiffusionXLPipeline
                    logger.info("Using StableDiffusionXLPipeline for SDXL Turbo model (Turbo pipeline not available)")
            elif "xl" in self.model_name.lower() and "turbo" not in self.model_name.lower():
                # Regular SDXL
                pipeline_class = StableDiffusionXLPipeline
                logger.info("Using StableDiffusionXLPipeline for SDXL model")
            else:
                # Standard Stable Diffusion (v1.x, v2.x)
                pipeline_class = StableDiffusionPipeline
                logger.info("Using StableDiffusionPipeline for standard model")
            
            # Load pipeline with memory optimizations
            self.pipeline = pipeline_class.from_pretrained(
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
            self.registry = PackRegistry('packs/config.json')
            
            # Try to load image-focused packs if available
            try:
                from neuromod.pack_system import Pack
                
                with open('packs/image_focused_packs.json', 'r') as f:
                    image_packs = json.load(f)
                    # Convert to Pack objects and merge into registry
                    for pack_name, pack_data in image_packs['packs'].items():
                        try:
                            pack_obj = Pack.from_dict(pack_data)
                            self.registry.packs[pack_name] = pack_obj
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
            self.neuromod_tool = NeuromodTool(self.registry, dummy_model, dummy_tokenizer)
            
            pack_count = len(self.get_available_packs())
            logger.info(f"Neuromodulation system initialized with {pack_count} packs")
            
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
        
        # Check if this is a Turbo model (uses different defaults)
        is_turbo = "turbo" in self.model_name.lower()
        
        # Default generation parameters (Turbo models use fewer steps and no guidance)
        if is_turbo:
            base_params = {
                'num_inference_steps': 1,  # Turbo optimized for 1-4 steps
                'strength': 1.0
            }
        else:
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
                        default_steps = 1 if is_turbo else 50
                        base_val = base_params.get(key, default_steps)
                        current_val = self.generation_params[key]
                        increase = current_val - base_val
                        self.generation_params[key] = int(base_val + (increase * intensity))
                        # For Turbo, clamp to reasonable range (1-10 steps)
                        if is_turbo:
                            self.generation_params[key] = max(1, min(10, self.generation_params[key]))
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
        if not is_turbo and 'guidance_scale' in self.generation_params:
            self.generation_params['guidance_scale'] = max(1.0, min(20.0, self.generation_params['guidance_scale']))
        
        if 'num_inference_steps' in self.generation_params:
            if is_turbo:
                self.generation_params['num_inference_steps'] = max(1, min(10, self.generation_params['num_inference_steps']))
            else:
                self.generation_params['num_inference_steps'] = max(10, min(100, self.generation_params['num_inference_steps']))
        
        if 'strength' in self.generation_params:
            self.generation_params['strength'] = max(0.1, min(2.0, self.generation_params['strength']))
        
        logger.info(f"Adapted generation parameters: {self.generation_params}")
    
    def generate_image(self, prompt: str, pack_name: Optional[str] = None, 
                      intensity: float = 0.5, **kwargs) -> Dict[str, Any]:
        """
        Generate an image with optional neuromodulation effects.
        
        This method now captures both the final image and the raw latents
        (pre-VAE decoder) for frequency domain analysis.
        
        Args:
            prompt: Text prompt for image generation
            pack_name: Name of neuromodulation pack to apply
            intensity: Intensity of the neuromodulation effects
            **kwargs: Additional generation parameters
        
        Returns:
            Dictionary containing the generated image, raw latents, and metadata
        """
        try:
            start_time = time.time()
            
            # Check if this is a Turbo model (uses different defaults)
            is_turbo = "turbo" in self.model_name.lower()
            
            # Apply neuromodulation if specified
            if pack_name:
                success = self.apply_neuromodulation_pack(pack_name, intensity)
                if not success:
                    logger.warning(f"Failed to apply pack {pack_name}, using default parameters")
            
            # Get generation parameters (either from neuromodulation or defaults)
            # Turbo models use fewer steps and no guidance scale
            if is_turbo:
                default_params = {
                    'num_inference_steps': 1,  # Turbo models are optimized for 1-4 steps
                    'guidance_scale': 0.0,  # Turbo models don't use guidance scale
                }
            else:
                default_params = {
                    'guidance_scale': 7.5,
                    'num_inference_steps': 50,
                    'eta': 0.0
                }
            
            gen_params = getattr(self, 'generation_params', default_params)
            
            # For Turbo models, ensure we don't use guidance_scale
            if is_turbo:
                gen_params.pop('guidance_scale', None)
                # Ensure steps are reasonable for Turbo (1-4 is typical)
                if 'num_inference_steps' in gen_params:
                    gen_params['num_inference_steps'] = max(1, min(10, gen_params['num_inference_steps']))
            
            # Override with any provided kwargs
            gen_params.update(kwargs)
            
            logger.info(f"Generating image with prompt: '{prompt}'")
            logger.info(f"Generation parameters: {gen_params}")
            
            # Don't add visual prompts - the effects should only modify generation behavior
            # The model should generate the normal prompt, but with modified internal processing
            enhanced_prompt = prompt
            
            # Generate the image and capture latents
            with torch.no_grad():
                # Build pipeline call arguments (without prompt, we'll pass it explicitly)
                pipeline_kwargs = {
                    **{k: v for k, v in gen_params.items() if k not in ['color_prompts', 'style_prompts', 'composition_prompts', 'detail_prompts', 'synesthetic_prompts', 'motion_prompts']}
                }
                
                # Only add guidance_scale if not a Turbo model
                if not is_turbo and 'guidance_scale' in gen_params:
                    pipeline_kwargs['guidance_scale'] = gen_params['guidance_scale']
                
                # Only add eta if not a Turbo model
                if not is_turbo and 'eta' in gen_params:
                    pipeline_kwargs['eta'] = gen_params['eta']
                
                # 1. Generate Latents (Pre-Convolution)
                # We use output_type="latent" to get the raw VAE input
                try:
                    output_latents = self.pipeline(
                        prompt=enhanced_prompt,
                        output_type="latent",
                        **pipeline_kwargs
                    ).images  # This is a tensor [1, 4, 64, 64] or [1, 4, H, W]
                    
                    # 2. Manually Decode to Image (Un-Convolution)
                    # Scale latents as required by SD VAE
                    # Different models may have different scaling factors
                    if hasattr(self.pipeline.vae.config, 'scaling_factor'):
                        scaling_factor = self.pipeline.vae.config.scaling_factor
                    else:
                        # Default scaling factor for most SD models
                        scaling_factor = 0.18215
                    
                    latents_scaled = output_latents / scaling_factor
                    
                    # Decode using VAE
                    image_tensor = self.pipeline.vae.decode(latents_scaled, return_dict=False)[0]
                    
                    # Post-process to PIL image
                    if hasattr(self.pipeline, 'image_processor'):
                        image = self.pipeline.image_processor.postprocess(image_tensor, output_type="pil")[0]
                    else:
                        # Fallback for older pipeline versions
                        image = self.pipeline.numpy_to_pil(image_tensor.cpu().permute(0, 2, 3, 1).numpy())[0]
                    
                except Exception as e:
                    # Fallback to standard generation if latent capture fails
                    logger.warning(f"Failed to capture latents, using standard generation: {e}")
                    # Add prompt back for fallback generation
                    pipeline_kwargs['prompt'] = enhanced_prompt
                    result = self.pipeline(**pipeline_kwargs)
                    image = result.images[0] if result.images else None
                    output_latents = None
            
            generation_time = time.time() - start_time
            
            if image is None:
                raise RuntimeError("No image generated")
            
            # Prepare response
            response = {
                'image': image,
                'latents': output_latents,  # Return raw latents for frequency analysis
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
                'latents': None,
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


def demo_lsd_image_generation(model_name: str = None):
    """Demo function to generate images with LSD neuromodulation and frequency analysis"""
    
    print("🎨 LSD Image Generation Demo with Frequency Analysis")
    print("=" * 60)
    
    # Ensure output directory exists
    output_dir = "outputs/reports/test_suite"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the image generation interface
    try:
        if model_name:
            image_gen = ImageNeuromodInterface(model_name=model_name)
        else:
            image_gen = ImageNeuromodInterface()
        print(f"✅ Image generation interface initialized (model: {image_gen.model_name})")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Test prompts for normal, everyday subjects (to see LSD effects)
    # Patterns often show clear frequency signatures
    test_prompts = [
        "A peaceful garden with flowers and trees",
        "A simple geometric pattern on a wall"
    ]
    
    # Generate images with and without LSD pack
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n--- Test {i}: {prompt[:50]}... ---")
        
        # Generate without neuromodulation (baseline)
        print("🔄 Generating baseline image...")
        baseline_result = image_gen.generate_image(prompt)
        
        if baseline_result['success']:
            baseline_image = baseline_result['image']
            baseline_filename = f"{output_dir}/test_{i}_baseline.png"
            baseline_image.save(baseline_filename)
            print(f"✅ Baseline image saved: {baseline_filename}")
            
            # Generate frequency analysis for baseline
            if baseline_result.get('latents') is not None:
                base_analysis_path = f"{output_dir}/test_{i}_baseline_analysis.png"
                FrequencyAnalyzer.save_spectral_analysis(
                    baseline_image,
                    baseline_result['latents'],
                    f"Baseline: {prompt}",
                    base_analysis_path
                )
                print(f"✅ Baseline frequency analysis saved: {base_analysis_path}")
            else:
                print("⚠️  No latents captured for baseline (frequency analysis skipped)")
        else:
            print(f"❌ Baseline generation failed: {baseline_result.get('error', 'Unknown error')}")
            continue
        
        # Generate with LSD pack
        print("🔄 Generating with LSD neuromodulation...")
        lsd_result = image_gen.generate_image(
            prompt=prompt,
            pack_name="lsd",
            intensity=0.8
        )
        
        if lsd_result['success']:
            lsd_image = lsd_result['image']
            lsd_filename = f"{output_dir}/test_{i}_lsd.png"
            lsd_image.save(lsd_filename)
            print(f"✅ LSD image saved: {lsd_filename}")
            
            # Generate frequency analysis for LSD
            if lsd_result.get('latents') is not None:
                lsd_analysis_path = f"{output_dir}/test_{i}_lsd_analysis.png"
                FrequencyAnalyzer.save_spectral_analysis(
                    lsd_image,
                    lsd_result['latents'],
                    f"LSD (Int=0.8): {prompt}",
                    lsd_analysis_path
                )
                print(f"✅ LSD frequency analysis saved: {lsd_analysis_path}")
            else:
                print("⚠️  No latents captured for LSD (frequency analysis skipped)")
            
            # Show generation parameters
            params = lsd_result['generation_params']
            print(f"   Generation time: {lsd_result['generation_time']:.2f}s")
            if 'guidance_scale' in params:
                print(f"   Guidance scale: {params['guidance_scale']:.2f}")
            print(f"   Inference steps: {params.get('num_inference_steps', 'N/A')}")
        else:
            print(f"❌ LSD generation failed: {lsd_result.get('error', 'Unknown error')}")
        
        print("-" * 60)
    
    print("\n🎉 Demo completed! Check the generated images and frequency analyses.")


def _save_image_with_analysis(result, base_filename, pack_name=None):
    """
    Helper function to save an image and generate frequency analysis if latents are available.
    
    Args:
        result: Result dictionary from generate_image()
        base_filename: Base filename for the image (without extension)
        pack_name: Optional pack name for the analysis title
    """
    if not result['success']:
        return None
    
    os.makedirs("outputs/reports/test_suite", exist_ok=True)
    
    # Save the image
    image_filename = f"{base_filename}.png"
    result['image'].save(image_filename)
    print(f"✅ Image saved: {image_filename}")
    
    # Generate frequency analysis if latents are available
    if result.get('latents') is not None:
        analysis_filename = f"{base_filename}_analysis.png"
        title = f"{pack_name + ': ' if pack_name else ''}{result['prompt']}"
        try:
            FrequencyAnalyzer.save_spectral_analysis(
                result['image'],
                result['latents'],
                title,
                analysis_filename
            )
            print(f"✅ Frequency analysis saved: {analysis_filename}")
        except Exception as e:
            logger.warning(f"Failed to generate frequency analysis: {e}")
    else:
        print("⚠️  No latents captured (frequency analysis skipped)")
    
    return image_filename


def interactive_image_generation(model_name: str = None):
    """Interactive image generation with neuromodulation"""
    
    print("🎨 Interactive Image Generation with Neuromodulation")
    print("=" * 60)
    
    # Initialize the interface
    try:
        if model_name:
            image_gen = ImageNeuromodInterface(model_name=model_name)
            print(f"✅ Ready for image generation (model: {image_gen.model_name})")
        else:
            image_gen = ImageNeuromodInterface()
            print(f"✅ Ready for image generation (model: {image_gen.model_name})")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Get available packs
    available_packs = image_gen.get_available_packs()
    pack_categories = image_gen.list_packs_by_category()
    
    # Track current settings
    current_intensity = 0.7
    current_params = {}
    is_turbo = "turbo" in image_gen.model_name.lower()
    
    print("\nAvailable commands:")
    print("  <prompt> - Generate image with default settings")
    print("  <pack_name> <prompt> - Generate with pack (e.g., 'lsd a cat')")
    print("  /pack <pack_name> <prompt> - Generate with specific pack")
    print("  /pack <pack_name> <intensity> <prompt> - Generate with pack and custom intensity (0.0-1.0)")
    print("  /params <key>=<value> ... - Set generation parameters (steps, guidance, eta, strength)")
    print("  /intensity <value> - Set default intensity for pack applications (0.0-1.0)")
    print("  /settings - Show current generation settings")
    print("  /list - List all available packs")
    print("  /categories - List packs by category")
    print("  /clear - Clear neuromodulation effects and reset parameters")
    print("  /model <model_name> - Change model (requires restart)")
    print("  /help - Show this help message")
    print("  /quit - Exit")
    print()
    print("💡 Try normal, everyday prompts to see how neuromodulation affects perception!")
    print("   Examples: 'a cat sitting on a chair', 'a cup of coffee', 'a tree in a park'")
    print()
    print(f"📦 {len(available_packs)} packs available. Use /list to see all packs.")
    print()
    
    while True:
        try:
            user_input = input("🎨 Enter command or prompt: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                break
            elif user_input.lower() == '/help':
                print("\n📖 Help:")
                print("  Generation Parameters:")
                print("    - steps=<N> - Number of inference steps (1-100, Turbo: 1-10)")
                print("    - guidance=<N> - Guidance scale (1.0-20.0, Turbo: not used)")
                print("    - eta=<N> - ETA parameter (0.0-1.0, Turbo: not used)")
                print("    - strength=<N> - Strength parameter (0.1-2.0)")
                print("  Examples:")
                print("    /params steps=30 guidance=10.0")
                print("    /pack lsd 0.9 a cat")
                print("    /intensity 0.5")
                print()
                continue
            elif user_input.lower() == '/clear':
                image_gen.clear_neuromodulation()
                current_intensity = 0.7
                current_params = {}
                print("✅ Neuromodulation effects and parameters cleared")
                continue
            elif user_input.lower() == '/settings':
                print("\n⚙️  Current Settings:")
                print(f"   Model: {image_gen.model_name}")
                print(f"   Default Intensity: {current_intensity}")
                if current_params:
                    print("   Custom Parameters:")
                    for key, value in current_params.items():
                        print(f"      {key}: {value}")
                else:
                    print("   Custom Parameters: None (using defaults)")
                # Show model-specific defaults
                if is_turbo:
                    print("   Model Type: Turbo (optimized for 1-4 steps, no guidance scale)")
                else:
                    print("   Model Type: Standard (guidance_scale=7.5, steps=50)")
                print()
                continue
            elif user_input.lower().startswith('/intensity '):
                try:
                    intensity_val = float(user_input.split()[1])
                    if 0.0 <= intensity_val <= 1.0:
                        current_intensity = intensity_val
                        print(f"✅ Default intensity set to {current_intensity}")
                    else:
                        print("❌ Intensity must be between 0.0 and 1.0")
                except (ValueError, IndexError):
                    print("❌ Usage: /intensity <value> (0.0-1.0)")
                continue
            elif user_input.lower().startswith('/params '):
                try:
                    params_str = user_input[8:].strip()
                    params_dict = {}
                    for param in params_str.split():
                        if '=' in param:
                            key, value = param.split('=', 1)
                            try:
                                # Try to parse as float first, then int
                                if '.' in value:
                                    params_dict[key] = float(value)
                                else:
                                    params_dict[key] = int(value)
                            except ValueError:
                                print(f"⚠️  Invalid value for {key}: {value}, skipping")
                    
                    # Validate parameters
                    if 'steps' in params_dict:
                        if is_turbo:
                            params_dict['steps'] = max(1, min(10, params_dict['steps']))
                        else:
                            params_dict['steps'] = max(10, min(100, params_dict['steps']))
                        params_dict['num_inference_steps'] = params_dict.pop('steps')
                    
                    if 'guidance' in params_dict:
                        if is_turbo:
                            print("⚠️  Guidance scale not used for Turbo models, ignoring")
                        else:
                            params_dict['guidance_scale'] = max(1.0, min(20.0, params_dict.pop('guidance')))
                    
                    if 'eta' in params_dict:
                        if is_turbo:
                            print("⚠️  ETA not used for Turbo models, ignoring")
                        else:
                            params_dict['eta'] = max(0.0, min(1.0, params_dict['eta']))
                    
                    if 'strength' in params_dict:
                        params_dict['strength'] = max(0.1, min(2.0, params_dict['strength']))
                    
                    current_params.update(params_dict)
                    print(f"✅ Parameters updated: {current_params}")
                except Exception as e:
                    print(f"❌ Error parsing parameters: {e}")
                    print("   Usage: /params steps=30 guidance=10.0 eta=0.5")
                continue
            elif user_input.lower() == '/list':
                print("\n📦 Available Packs:")
                for pack in sorted(available_packs):
                    print(f"   • {pack}")
                print()
                continue
            elif user_input.lower() == '/categories':
                print("\n📦 Packs by Category:")
                for category, packs in pack_categories.items():
                    print(f"\n   {category.upper()}:")
                    for pack in sorted(packs):
                        print(f"      • {pack}")
                print()
                continue
            elif user_input.startswith('/pack '):
                parts = user_input[6:].strip().split()
                if len(parts) >= 2:
                    pack_name = parts[0]
                    # Check if second part is a number (intensity)
                    try:
                        intensity = float(parts[1])
                        if 0.0 <= intensity <= 1.0:
                            # Format: /pack <pack_name> <intensity> <prompt>
                            prompt = ' '.join(parts[2:]) if len(parts) > 2 else ''
                            if not prompt:
                                print("❌ Usage: /pack <pack_name> <intensity> <prompt>")
                                continue
                        else:
                            # Not a valid intensity, treat as prompt
                            intensity = current_intensity
                            prompt = ' '.join(parts[1:])
                    except ValueError:
                        # Not a number, treat as prompt
                        intensity = current_intensity
                        prompt = ' '.join(parts[1:])
                    
                    if pack_name in available_packs:
                        print(f"🔄 Generating with {pack_name} pack (intensity: {intensity}): '{prompt}'")
                        # Merge current_params with generation
                        gen_kwargs = current_params.copy()
                        result = image_gen.generate_image(prompt, pack_name=pack_name, intensity=intensity, **gen_kwargs)
                        if result['success']:
                            base_filename = f"outputs/reports/test_suite/{pack_name}_{int(time.time())}"
                            _save_image_with_analysis(result, base_filename, pack_name=pack_name)
                            print(f"   Generation time: {result['generation_time']:.2f}s")
                            print(f"   Parameters: {result['generation_params']}")
                        else:
                            print(f"❌ Generation failed: {result.get('error')}")
                    else:
                        print(f"❌ Pack '{pack_name}' not found. Use /list to see available packs.")
                else:
                    print("❌ Usage: /pack <pack_name> [intensity] <prompt>")
                continue
            elif user_input.startswith('/model '):
                new_model = user_input[7:].strip()
                print(f"ℹ️  Model change requires restart. Current model: {image_gen.model_name}")
                print(f"   To use '{new_model}', restart with: python demo/image_generation_demo.py --model {new_model}")
                continue
            else:
                # Check if input starts with a pack name (common packs)
                # Try to detect if user wants to use a pack
                words = user_input.split()
                if len(words) > 1 and words[0] in available_packs:
                    pack_name = words[0]
                    prompt = ' '.join(words[1:])
                    print(f"🔄 Generating with {pack_name} pack (intensity: {current_intensity}): '{prompt}'")
                    gen_kwargs = current_params.copy()
                    result = image_gen.generate_image(prompt, pack_name=pack_name, intensity=current_intensity, **gen_kwargs)
                    if result['success']:
                        base_filename = f"outputs/reports/test_suite/{pack_name}_{int(time.time())}"
                        _save_image_with_analysis(result, base_filename, pack_name=pack_name)
                        print(f"   Generation time: {result['generation_time']:.2f}s")
                        print(f"   Parameters: {result['generation_params']}")
                    else:
                        print(f"❌ Generation failed: {result.get('error')}")
                else:
                    # Default generation without neuromodulation
                    print(f"🔄 Generating: '{user_input}'")
                    gen_kwargs = current_params.copy()
                    result = image_gen.generate_image(user_input, **gen_kwargs)
                    if result['success']:
                        base_filename = f"outputs/reports/test_suite/baseline_{int(time.time())}"
                        _save_image_with_analysis(result, base_filename)
                        print(f"   Generation time: {result['generation_time']:.2f}s")
                        print(f"   Parameters: {result['generation_params']}")
                    else:
                        print(f"❌ Generation failed: {result.get('error')}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Image Generation Demo with Neuromodulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with default model
  python demo/image_generation_demo.py
  
  # Interactive mode with specific model
  python demo/image_generation_demo.py --model sd-xl
  
  # Interactive mode with custom model
  python demo/image_generation_demo.py --model stabilityai/stable-diffusion-xl-base-1.0
  
  # Run demo script
  python demo/image_generation_demo.py --demo
  
  # Run demo with specific model
  python demo/image_generation_demo.py --demo --model sd-v2-1

Available model shortcuts:
  sd-v1-5     - runwayml/stable-diffusion-v1-5 (default)
  sd-v1-4     - CompVis/stable-diffusion-v1-4
  sd-v2-1     - stabilityai/stable-diffusion-2-1
  sd-xl       - stabilityai/stable-diffusion-xl-base-1.0
  sdxl-turbo  - stabilityai/sdxl-turbo (fast, 1-4 steps)
        """
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Stable Diffusion model to use (shortcut or full HuggingFace path)'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run the LSD demo instead of interactive mode'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model shortcuts and exit'
    )
    parser.add_argument(
        '--list-packs',
        action='store_true',
        help='List available packs and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_models:
        print("Available model shortcuts:")
        for shortcut, model_path in COMMON_MODELS.items():
            print(f"  {shortcut:12} -> {model_path}")
        print("\nYou can also use any HuggingFace Stable Diffusion model path directly.")
        exit(0)
    
    if args.list_packs:
        try:
            image_gen = ImageNeuromodInterface()
            packs = image_gen.get_available_packs()
            categories = image_gen.list_packs_by_category()
            print(f"\n📦 Available Packs ({len(packs)} total):\n")
            for category, pack_list in categories.items():
                print(f"{category.upper()}:")
                for pack in sorted(pack_list):
                    print(f"  • {pack}")
                print()
        except Exception as e:
            print(f"❌ Failed to load packs: {e}")
        exit(0)
    
    if args.demo:
        demo_lsd_image_generation(model_name=args.model)
    else:
        interactive_image_generation(model_name=args.model)
