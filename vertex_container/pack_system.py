"""
Modular pack system using base-level effects with JSON configuration
Each pack defines a structured set of effects with weights and directions
"""

import json
import os
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .effects import EffectRegistry, BaseEffect

@dataclass
class EffectConfig:
    """Configuration for a single effect"""
    effect: str
    weight: float = 0.5
    direction: str = "up"
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EffectConfig':
        """Create EffectConfig from dictionary"""
        return cls(
            effect=data['effect'],
            weight=data.get('weight', 0.5),
            direction=data.get('direction', 'up'),
            parameters=data.get('parameters', {})
        )

@dataclass
class Pack:
    """A neuromodulation pack with multiple effects"""
    name: str
    description: str
    effects: List[EffectConfig]
    
    def __post_init__(self):
        # Validate effect configurations
        for effect_config in self.effects:
            if not isinstance(effect_config, EffectConfig):
                raise ValueError(f"Effect must be EffectConfig, got {type(effect_config)}")
            if effect_config.weight < 0.0 or effect_config.weight > 1.0:
                raise ValueError(f"Effect weight must be 0.0-1.0, got {effect_config.weight}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pack':
        """Create Pack from dictionary"""
        effects = [EffectConfig.from_dict(effect_data) for effect_data in data['effects']]
        return cls(
            name=data['name'],
            description=data['description'],
            effects=effects
        )

class PackManager:
    """Manages the application of packs using modular effects"""
    
    def __init__(self):
        self.effect_registry = EffectRegistry()
        self.active_effects: List[BaseEffect] = []
        
    def apply_pack(self, pack: Pack, model) -> Dict[str, Any]:
        """Apply a pack to the model"""
        results = {
            "applied_effects": [],
            "logits_processors": [],
            "errors": []
        }
        
        # Clear any existing effects
        self.clear_effects()
        
        # Apply each effect in the pack
        for effect_config in pack.effects:
            try:
                # Get effect parameters
                params = effect_config.parameters or {}
                params.update({
                    "weight": effect_config.weight,
                    "direction": effect_config.direction
                })
                
                # Create and apply the effect
                effect = self.effect_registry.get_effect(effect_config.effect, **params)
                effect.apply(model, **params)
                
                # Store active effect
                self.active_effects.append(effect)
                
                # Get logits processor if available (only sampler effects have this)
                if hasattr(effect, 'get_logits_processor'):
                    logits_processor = effect.get_logits_processor()
                    if logits_processor:
                        results["logits_processors"].append(logits_processor)
                
                results["applied_effects"].append({
                    "effect": effect_config.effect,
                    "weight": effect_config.weight,
                    "direction": effect_config.direction
                })
                
            except Exception as e:
                results["errors"].append({
                    "effect": effect_config.effect,
                    "error": str(e)
                })
                
        return results
    
    def get_logits_processors(self) -> List:
        """Get all active logits processors"""
        processors = []
        for effect in self.active_effects:
            processor = effect.get_logits_processor()
            if processor:
                processors.append(processor)
        return processors
    
    def apply_steering(self, hidden_states) -> torch.Tensor:
        """Apply steering effects to hidden states"""
        for effect in self.active_effects:
            if hasattr(effect, 'apply_steering'):
                hidden_states = effect.apply_steering(hidden_states)
        return hidden_states
    
    def modify_kv_cache(self, kv_cache):
        """Apply KV cache modifications"""
        for effect in self.active_effects:
            if hasattr(effect, 'modify_kv_cache'):
                kv_cache = effect.modify_kv_cache(kv_cache)
        return kv_cache
    
    def clear_effects(self):
        """Clear all active effects"""
        for effect in self.active_effects:
            try:
                effect.cleanup()
            except Exception as e:
                print(f"Warning: Error cleaning up effect {type(effect).__name__}: {e}")
        self.active_effects.clear()
    
    def get_effect_info(self) -> Dict[str, Any]:
        """Get information about available effects"""
        return {
            "available_effects": self.effect_registry.list_effects(),
            "active_effects": [
                {
                    "type": type(effect).__name__,
                    "weight": getattr(effect, 'weight', None),
                    "direction": getattr(effect, 'direction', None)
                }
                for effect in self.active_effects
            ]
        }

# ============================================================================
# LEGACY PACK CREATION FUNCTIONS (DEPRECATED - USE JSON CONFIG)
# ============================================================================

# These functions are kept for backward compatibility but are deprecated.

# ============================================================================
# PACK REGISTRY
# ============================================================================

class PackRegistry:
    """Registry for all available packs loaded from JSON configuration"""
    
    def __init__(self, config_path: str = "packs/config.json"):
        self.config_path = config_path
        self.packs: Dict[str, Pack] = {}
        self.load_packs()
    
    def load_packs(self):
        """Load packs from JSON configuration file"""
        try:
            if not os.path.exists(self.config_path):
                print(f"Warning: Config file not found at {self.config_path}")
                return
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for pack_name, pack_data in config.get('packs', {}).items():
                try:
                    pack = Pack.from_dict(pack_data)
                    self.packs[pack_name] = pack
                except Exception as e:
                    print(f"Warning: Failed to load pack {pack_name}: {e}")
                    
            print(f"Loaded {len(self.packs)} packs from {self.config_path}")
            
        except Exception as e:
            print(f"Error loading packs from {self.config_path}: {e}")
    
    def reload_packs(self):
        """Reload packs from configuration file"""
        self.packs.clear()
        self.load_packs()
    
    def get_pack(self, pack_name: str) -> Pack:
        """Get a pack by name"""
        if pack_name not in self.packs:
            raise ValueError(f"Unknown pack: {pack_name}")
        return self.packs[pack_name]
        
    def list_packs(self) -> List[str]:
        """List all available packs"""
        return list(self.packs.keys())
    
    def get_pack_info(self, pack_name: str) -> Dict[str, Any]:
        """Get detailed information about a pack"""
        if pack_name not in self.packs:
            raise ValueError(f"Unknown pack: {pack_name}")
        
        pack = self.packs[pack_name]
        return {
            "name": pack.name,
            "description": pack.description,
            "effects": [
                {
                    "effect": effect.effect,
                    "weight": effect.weight,
                    "direction": effect.direction,
                    "parameters": effect.parameters
                }
                for effect in pack.effects
            ]
        }
        
    def add_pack(self, pack: Pack):
        """Add a new pack to the registry"""
        self.packs[pack.name] = pack
        
    def remove_pack(self, pack_name: str):
        """Remove a pack from the registry"""
        if pack_name in self.packs:
            del self.packs[pack_name]
    
    def save_packs_to_json(self, output_path: str = None):
        """Save current packs to JSON file"""
        if output_path is None:
            output_path = self.config_path
        
        config = {
            "packs": {
                pack_name: {
                    "name": pack.name,
                    "description": pack.description,
                    "effects": [
                        {
                            "effect": effect.effect,
                            "weight": effect.weight,
                            "direction": effect.direction,
                            "parameters": effect.parameters
                        }
                        for effect in pack.effects
                    ]
                }
                for pack_name, pack in self.packs.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved {len(self.packs)} packs to {output_path}")
