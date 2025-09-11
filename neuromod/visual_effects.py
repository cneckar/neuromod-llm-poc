"""
Visual neuromodulation effects for image generation models
These effects are specifically designed to work with Stable Diffusion and similar models
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import math
import random

class VisualEffect(ABC):
    """Base class for visual neuromodulation effects"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None, **kwargs):
        self.weight = max(0.0, min(1.0, weight))
        self.direction = direction
        # Combine parameters dict with any additional kwargs
        self.parameters = parameters or {}
        self.parameters.update(kwargs)
        
    @abstractmethod
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply the effect to image generation parameters"""
        pass
    
    def apply(self, model, **kwargs):
        """Apply the effect to a model (compatibility with base effect system)"""
        # For visual effects, we don't actually modify the model
        # The effects are applied during image generation
        pass
        
    def cleanup(self):
        """Clean up the effect (compatibility with base effect system)"""
        # Visual effects don't need cleanup
        pass
        
    def get_effective_value(self, base_value: float, max_change: float) -> float:
        """Calculate effective value based on weight and direction"""
        if self.direction == "up":
            return base_value + (max_change * self.weight)
        elif self.direction == "down":
            return base_value - (max_change * self.weight)
        else:  # neutral
            return base_value

class ColorBiasEffect(VisualEffect):
    """Effect that biases color generation towards specific palettes and characteristics"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply color bias to generation parameters"""
        palette = self.parameters.get("color_palette", "default")
        
        # Modify guidance scale based on color intensity
        if palette in ["neon_cyberpunk", "psychedelic_flow"]:
            generation_params["guidance_scale"] *= (1.0 + self.weight * 0.3)
            
        # Don't add prompts - just modify generation behavior
        # The effects should influence HOW the model generates, not WHAT it generates
            
        # Adjust noise level for color effects
        if self.parameters.get("saturation_boost", 0) > 0:
            generation_params["eta"] = min(1.0, generation_params.get("eta", 0.0) + self.weight * 0.2)
            
        return generation_params
        
    def _get_color_prompts(self, palette: str) -> Optional[str]:
        """Get color-specific prompt additions"""
        color_mappings = {
            "neon_cyberpunk": "neon colors, cyberpunk aesthetic, high saturation, electric blues and pinks",
            "psychedelic_flow": "psychedelic colors, flowing hues, rainbow gradients, vibrant saturation",
            "enhanced_natural": "enhanced natural colors, warm tones, rich earth colors, natural saturation",
            "sacred_geometric": "spiritual colors, sacred geometry palette, mystical hues, geometric patterns"
        }
        return color_mappings.get(palette)

class StyleTransferEffect(VisualEffect):
    """Effect that influences the style and texture of generated images"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply style transfer to generation parameters"""
        style = self.parameters.get("style", "default")
        
        # Modify inference steps based on style complexity
        if style in ["fractal_geometric", "sacred_geometry"]:
            generation_params["num_inference_steps"] = int(
                generation_params.get("num_inference_steps", 50) * (1.0 + self.weight * 0.4)
            )
            
        # Don't add prompts - just modify generation behavior
            
        # Adjust guidance for complex styles
        if style in ["fractal_geometric", "organic_flowing"]:
            generation_params["guidance_scale"] *= (1.0 + self.weight * 0.2)
            
        return generation_params
        
    def _get_style_prompts(self, style: str) -> Optional[str]:
        """Get style-specific prompt additions"""
        style_mappings = {
            "fractal_geometric": "fractal patterns, geometric structures, mathematical beauty, intricate details",
            "organic_flowing": "organic flowing patterns, natural curves, smooth transitions, fluid forms",
            "organic_natural": "natural patterns, organic textures, fractal nature, breathing forms",
            "sacred_geometry": "sacred geometry, spiritual patterns, geometric harmony, mystical symbols"
        }
        return style_mappings.get(style)

class CompositionBiasEffect(VisualEffect):
    """Effect that influences the composition and structure of generated images"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply composition bias to generation parameters"""
        composition_type = self.parameters.get("composition_type", "default")
        
        # Don't add prompts - just modify generation behavior
            
        # Adjust parameters based on composition type
        if composition_type in ["radial_symmetry", "sacred_geometry"]:
            generation_params["guidance_scale"] *= (1.0 + self.weight * 0.15)
            
        return generation_params
        
    def _get_composition_prompts(self, composition_type: str) -> Optional[str]:
        """Get composition-specific prompt additions"""
        comp_mappings = {
            "radial_symmetry": "radial symmetry, centered composition, balanced layout",
            "flowing_organic": "flowing composition, organic curves, natural balance",
            "natural_harmony": "natural harmony, balanced composition, organic flow",
            "sacred_geometry": "sacred geometry composition, golden ratio, spiritual balance"
        }
        return comp_mappings.get(composition_type)

class VisualEntropyEffect(VisualEffect):
    """Effect that increases visual complexity and detail"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply visual entropy to generation parameters"""
        
        # Increase inference steps for more detail
        generation_params["num_inference_steps"] = int(
            generation_params.get("num_inference_steps", 50) * (1.0 + self.weight * 0.5)
        )
        
        # Don't add prompts - just modify generation behavior
        
        # Adjust guidance for detail preservation
        generation_params["guidance_scale"] *= (1.0 + self.weight * 0.2)
        
        return generation_params

class SynestheticMappingEffect(VisualEffect):
    """Effect that creates synesthetic color and pattern mappings"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply synesthetic mapping to generation parameters"""
        
        # Don't add prompts - just modify generation behavior
        
        # Increase temperature for more creative associations
        generation_params["temperature"] = generation_params.get("temperature", 1.0) * (1.0 + self.weight * 0.3)
        
        return generation_params

class MotionBlurEffect(VisualEffect):
    """Effect that simulates motion and flow in static images"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None):
        super().__init__(weight, direction, parameters)
        
    def apply_to_image_generation(self, generation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply motion blur to generation parameters"""
        
        # Don't add prompts - just modify generation behavior
        
        # Adjust parameters for motion effects
        generation_params["eta"] = min(1.0, generation_params.get("eta", 0.0) + self.weight * 0.15)
        
        return generation_params

# Factory function to create visual effects
def create_visual_effect(effect_type: str, weight: float = 0.5, direction: str = "up", parameters: Dict[str, Any] = None) -> VisualEffect:
    """Create a visual effect based on type"""
    effect_classes = {
        "color_bias": ColorBiasEffect,
        "style_transfer": StyleTransferEffect,
        "composition_bias": CompositionBiasEffect,
        "visual_entropy": VisualEntropyEffect,
        "synesthetic_mapping": SynestheticMappingEffect,
        "motion_blur": MotionBlurEffect
    }
    
    effect_class = effect_classes.get(effect_type)
    if effect_class:
        return effect_class(weight, direction, parameters)
    else:
        raise ValueError(f"Unknown visual effect type: {effect_type}")

# Function to apply visual effects to generation parameters
def apply_visual_effects_to_generation(effects: list, base_params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply a list of visual effects to generation parameters"""
    params = base_params.copy()
    
    for effect_data in effects:
        # Handle both EffectConfig objects and dictionaries
        if hasattr(effect_data, 'effect'):
            effect_type = effect_data.effect
            weight = effect_data.weight
            direction = effect_data.direction
            parameters = effect_data.parameters
        else:
            effect_type = effect_data.get("effect")
            weight = effect_data.get("weight", 0.5)
            direction = effect_data.get("direction", "up")
            parameters = effect_data.get("parameters", {})
        
        try:
            effect = create_visual_effect(effect_type, weight, direction, parameters)
            params = effect.apply_to_image_generation(params)
        except Exception as e:
            print(f"Warning: Could not apply visual effect {effect_type}: {e}")
            
    return params

# Function to combine prompts from visual effects
def combine_visual_prompts(generation_params: Dict[str, Any]) -> str:
    """Combine all visual effect prompts into a single prompt addition"""
    prompt_parts = []
    
    for key in ["color_prompts", "style_prompts", "composition_prompts", "detail_prompts", "synesthetic_prompts", "motion_prompts"]:
        if key in generation_params:
            prompt_parts.append(generation_params[key])
            
    return ", ".join(prompt_parts) if prompt_parts else ""
