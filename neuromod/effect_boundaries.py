#!/usr/bin/env python3
"""
Effect Boundaries and Type Enforcement System

This module enforces effect type boundaries and ensures proper application order
for the neuromodulation study.
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class EffectType(Enum):
    """Types of neuromodulation effects"""
    PROMPT_EFFECT = "prompt_effect"
    SAMPLING_EFFECT = "sampling_effect" 
    ACTIVATION_EFFECT = "activation_effect"
    OBJECTIVE_EFFECT = "objective_effect"

class BackendType(Enum):
    """Types of model backends"""
    HUGGINGFACE = "huggingface"  # Local models via HuggingFace Transformers
    VLLM = "vllm"                # Local models via vLLM for high throughput
    # Note: OpenAI/Anthropic APIs removed - they don't support activation effects

@dataclass
class EffectBoundary:
    """Defines boundaries for an effect type"""
    effect_type: EffectType
    allowed_backends: Set[BackendType]
    requires_model_access: bool
    application_order: int
    description: str

class EffectBoundaryEnforcer:
    """Enforces effect type boundaries and application order"""
    
    def __init__(self):
        self.boundaries = self._define_boundaries()
        self.application_order = self._define_application_order()
        
    def _define_boundaries(self) -> Dict[EffectType, EffectBoundary]:
        """Define boundaries for each effect type"""
        return {
            EffectType.PROMPT_EFFECT: EffectBoundary(
                effect_type=EffectType.PROMPT_EFFECT,
                allowed_backends={BackendType.HUGGINGFACE, BackendType.VLLM},
                requires_model_access=False,
                application_order=1,
                description="Effects that modify input prompts or context"
            ),
            EffectType.OBJECTIVE_EFFECT: EffectBoundary(
                effect_type=EffectType.OBJECTIVE_EFFECT,
                allowed_backends={BackendType.HUGGINGFACE, BackendType.VLLM},
                requires_model_access=True,
                application_order=2,
                description="Effects that modify model objectives or loss functions"
            ),
            EffectType.SAMPLING_EFFECT: EffectBoundary(
                effect_type=EffectType.SAMPLING_EFFECT,
                allowed_backends={BackendType.HUGGINGFACE, BackendType.VLLM},
                requires_model_access=False,
                application_order=3,
                description="Effects that modify sampling parameters (temperature, top_p, etc.)"
            ),
            EffectType.ACTIVATION_EFFECT: EffectBoundary(
                effect_type=EffectType.ACTIVATION_EFFECT,
                allowed_backends={BackendType.HUGGINGFACE, BackendType.VLLM},
                requires_model_access=True,
                application_order=4,
                description="Effects that modify model activations, attention, or hidden states"
            )
        }
    
    def _define_application_order(self) -> List[EffectType]:
        """Define the order in which effects should be applied"""
        return [
            EffectType.PROMPT_EFFECT,
            EffectType.OBJECTIVE_EFFECT,
            EffectType.SAMPLING_EFFECT,
            EffectType.ACTIVATION_EFFECT
        ]
    
    def get_effect_type(self, effect_name: str) -> EffectType:
        """Determine the type of an effect based on its name"""
        effect_type_mapping = {
            # Prompt effects
            "structured_prefaces": EffectType.PROMPT_EFFECT,
            "lexical_jitter": EffectType.PROMPT_EFFECT,
            
            # Objective effects
            "verifier_guided_decoding": EffectType.OBJECTIVE_EFFECT,
            "risk_preference_steering": EffectType.OBJECTIVE_EFFECT,
            
            # Sampling effects
            "temperature": EffectType.SAMPLING_EFFECT,
            "top_p": EffectType.SAMPLING_EFFECT,
            "frequency_penalty": EffectType.SAMPLING_EFFECT,
            "presence_penalty": EffectType.SAMPLING_EFFECT,
            "pulsed_sampler": EffectType.SAMPLING_EFFECT,
            "contrastive_decoding": EffectType.SAMPLING_EFFECT,
            "expert_mixing": EffectType.SAMPLING_EFFECT,
            "token_class_temperature": EffectType.SAMPLING_EFFECT,
            
            # Activation effects
            "attention_focus": EffectType.ACTIVATION_EFFECT,
            "attention_masking": EffectType.ACTIVATION_EFFECT,
            "qk_score_scaling": EffectType.ACTIVATION_EFFECT,
            "head_masking_dropout": EffectType.ACTIVATION_EFFECT,
            "head_reweighting": EffectType.ACTIVATION_EFFECT,
            "positional_bias_tweak": EffectType.ACTIVATION_EFFECT,
            "attention_oscillation": EffectType.ACTIVATION_EFFECT,
            "attention_sinks_anchors": EffectType.ACTIVATION_EFFECT,
            "steering": EffectType.ACTIVATION_EFFECT,
            "kv_decay": EffectType.ACTIVATION_EFFECT,
            "kv_compression": EffectType.ACTIVATION_EFFECT,
            "exponential_decay_kv": EffectType.ACTIVATION_EFFECT,
            "truncation_kv": EffectType.ACTIVATION_EFFECT,
            "stride_compression_kv": EffectType.ACTIVATION_EFFECT,
            "segment_gains_kv": EffectType.ACTIVATION_EFFECT,
            "router_temperature_bias": EffectType.ACTIVATION_EFFECT,
            "expert_masking_dropout": EffectType.ACTIVATION_EFFECT,
            "expert_persistence": EffectType.ACTIVATION_EFFECT,
            "style_affect_logit_bias": EffectType.SAMPLING_EFFECT,  # This affects logits
            "compute_at_test_scheduling": EffectType.ACTIVATION_EFFECT,
            "retrieval_rate_modulation": EffectType.ACTIVATION_EFFECT,
            "persona_voice_constraints": EffectType.PROMPT_EFFECT,
            "activation_additions": EffectType.ACTIVATION_EFFECT,
            "soft_projection": EffectType.ACTIVATION_EFFECT,
            "layer_wise_gain": EffectType.ACTIVATION_EFFECT,
            "noise_injection": EffectType.ACTIVATION_EFFECT,
            
            # Visual effects (treated as prompt effects for now)
            "color_bias": EffectType.PROMPT_EFFECT,
            "style_transfer": EffectType.PROMPT_EFFECT,
            "composition_bias": EffectType.PROMPT_EFFECT,
            "visual_entropy": EffectType.PROMPT_EFFECT,
            "synesthetic_mapping": EffectType.PROMPT_EFFECT,
            "motion_blur": EffectType.PROMPT_EFFECT
        }
        
        return effect_type_mapping.get(effect_name, EffectType.SAMPLING_EFFECT)
    
    def validate_effect_for_backend(self, effect_name: str, backend: BackendType) -> bool:
        """Validate that an effect can be used with a specific backend"""
        effect_type = self.get_effect_type(effect_name)
        boundary = self.boundaries[effect_type]
        
        return backend in boundary.allowed_backends
    
    def get_incompatible_effects(self, effects: List[str], backend: BackendType) -> List[str]:
        """Get list of effects that are incompatible with the backend"""
        incompatible = []
        
        for effect_name in effects:
            if not self.validate_effect_for_backend(effect_name, backend):
                incompatible.append(effect_name)
        
        return incompatible
    
    def sort_effects_by_application_order(self, effects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort effects by their application order"""
        def get_order(effect_dict):
            effect_name = effect_dict.get('effect', '')
            effect_type = self.get_effect_type(effect_name)
            return self.application_order.index(effect_type)
        
        return sorted(effects, key=get_order)
    
    def validate_pack_for_backend(self, pack: Dict[str, Any], backend: BackendType) -> Dict[str, Any]:
        """Validate a pack for use with a specific backend"""
        effects = pack.get('effects', [])
        effect_names = [effect.get('effect', '') for effect in effects]
        
        incompatible_effects = self.get_incompatible_effects(effect_names, backend)
        
        validation_result = {
            'valid': len(incompatible_effects) == 0,
            'backend': backend.value,
            'pack_name': pack.get('name', 'unknown'),
            'incompatible_effects': incompatible_effects,
            'total_effects': len(effects),
            'compatible_effects': len(effects) - len(incompatible_effects)
        }
        
        if incompatible_effects:
            validation_result['error'] = f"Pack contains {len(incompatible_effects)} effects incompatible with {backend.value} backend"
            validation_result['error_details'] = {
                'incompatible_effects': incompatible_effects,
                'reason': 'ActivationEffects require model access and are not supported by API backends'
            }
        
        return validation_result
    
    def enforce_effect_boundaries(self, pack: Dict[str, Any], backend: BackendType) -> Dict[str, Any]:
        """Enforce effect boundaries for a pack and backend"""
        validation = self.validate_pack_for_backend(pack, backend)
        
        if not validation['valid']:
            error_msg = f"Effect boundary violation: {validation['error']}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Sort effects by application order
        sorted_effects = self.sort_effects_by_application_order(pack.get('effects', []))
        
        return {
            'pack_name': pack.get('name', 'unknown'),
            'backend': backend.value,
            'effects': sorted_effects,
            'application_order': [self.get_effect_type(effect.get('effect', '')).value for effect in sorted_effects],
            'validation': validation
        }
    
    def log_effect_boundary_violation(self, pack_name: str, backend: BackendType, incompatible_effects: List[str]):
        """Log effect boundary violations for debugging"""
        logger.error(f"Effect boundary violation in pack '{pack_name}' for {backend.value} backend:")
        for effect in incompatible_effects:
            effect_type = self.get_effect_type(effect)
            boundary = self.boundaries[effect_type]
            logger.error(f"  - {effect} ({effect_type.value}) not allowed on {backend.value}")
            logger.error(f"    Allowed backends: {[b.value for b in boundary.allowed_backends]}")
    
    def get_backend_capabilities(self, backend: BackendType) -> Dict[str, Any]:
        """Get capabilities of a specific backend"""
        capabilities = {
            'backend': backend.value,
            'supports_prompt_effects': True,  # All backends support prompt effects
            'supports_sampling_effects': True,  # All backends support sampling effects
            'supports_activation_effects': backend in {BackendType.HUGGINGFACE, BackendType.VLLM},
            'supports_objective_effects': backend in {BackendType.HUGGINGFACE, BackendType.VLLM},
            'requires_model_access': backend in {BackendType.HUGGINGFACE, BackendType.VLLM}
        }
        
        return capabilities

def main():
    """Example usage of the effect boundary enforcer"""
    import json
    
    # Example pack
    example_pack = {
        "name": "test_pack",
        "description": "Test pack with mixed effects",
        "effects": [
            {"effect": "temperature", "weight": 0.5, "direction": "up"},
            {"effect": "attention_focus", "weight": 0.3, "direction": "up"},
            {"effect": "steering", "weight": 0.4, "direction": "up"},
            {"effect": "structured_prefaces", "weight": 0.2, "direction": "up"}
        ]
    }
    
    enforcer = EffectBoundaryEnforcer()
    
    # Test with different backends
    backends = [BackendType.HUGGINGFACE, BackendType.OPENAI, BackendType.VLLM]
    
    for backend in backends:
        print(f"\n--- Testing with {backend.value} backend ---")
        
        try:
            result = enforcer.enforce_effect_boundaries(example_pack, backend)
            print(f"✅ Pack validated successfully")
            print(f"   Effects: {len(result['effects'])}")
            print(f"   Application order: {result['application_order']}")
        except ValueError as e:
            print(f"❌ Pack validation failed: {e}")
        
        # Show capabilities
        capabilities = enforcer.get_backend_capabilities(backend)
        print(f"   Capabilities: {capabilities}")

if __name__ == "__main__":
    main()
