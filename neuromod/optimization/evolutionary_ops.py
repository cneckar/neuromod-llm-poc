"""
Evolutionary Operations for Pack Optimization

Implements mutation and crossover operations for evolutionary algorithms
in pack parameter optimization.
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import copy

from ..pack_system import Pack, PackRegistry

logger = logging.getLogger(__name__)

@dataclass
class MutationConfig:
    """Configuration for mutation operations"""
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    gaussian_noise_std: float = 0.05
    uniform_noise_range: float = 0.1
    adaptive_mutation: bool = True
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.5

@dataclass
class CrossoverConfig:
    """Configuration for crossover operations"""
    crossover_rate: float = 0.8
    crossover_type: str = 'uniform'  # 'uniform', 'single_point', 'two_point', 'arithmetic'
    blend_alpha: float = 0.5  # For arithmetic crossover
    tournament_size: int = 3

class PackMutator:
    """Handles mutation operations for pack parameters"""
    
    def __init__(self, config: MutationConfig = None):
        self.config = config or MutationConfig()
        self.pack_registry = PackRegistry()
    
    def mutate(self, pack: Pack, generation: int = 0) -> Pack:
        """
        Mutate a pack by modifying its parameters.
        
        Args:
            pack: Pack to mutate
            generation: Current generation (for adaptive mutation)
            
        Returns:
            Mutated pack
        """
        # Create a deep copy to avoid modifying the original
        mutated_pack = copy.deepcopy(pack)
        
        # Adaptive mutation rate
        mutation_rate = self._get_adaptive_mutation_rate(generation)
        
        # Mutate different aspects of the pack
        if random.random() < mutation_rate:
            self._mutate_effects(mutated_pack)
        
        if random.random() < mutation_rate:
            self._mutate_weights(mutated_pack)
        
        if random.random() < mutation_rate:
            self._mutate_parameters(mutated_pack)
        
        if random.random() < mutation_rate:
            self._mutate_metadata(mutated_pack)
        
        return mutated_pack
    
    def _get_adaptive_mutation_rate(self, generation: int) -> float:
        """Get adaptive mutation rate based on generation"""
        if not self.config.adaptive_mutation:
            return self.config.mutation_rate
        
        # Decrease mutation rate over generations
        decay_factor = max(0.1, 1.0 - (generation / 100.0))
        adaptive_rate = self.config.mutation_rate * decay_factor
        
        return np.clip(adaptive_rate, self.config.min_mutation_rate, self.config.max_mutation_rate)
    
    def _mutate_effects(self, pack: Pack):
        """Mutate the effects in the pack - can add, remove, or modify effects"""
        if not hasattr(pack, 'effects'):
            pack.effects = []
        
        # Add new effects with some probability
        if random.random() < self.config.mutation_rate * 0.3:  # 30% of mutation rate
            self._add_random_effect(pack)
        
        # Remove effects with some probability
        if len(pack.effects) > 1 and random.random() < self.config.mutation_rate * 0.2:  # 20% of mutation rate
            self._remove_random_effect(pack)
        
        # Modify existing effects
        for effect in pack.effects:
            if random.random() < self.config.mutation_rate:
                if hasattr(effect, 'intensity'):
                    # Add Gaussian noise to intensity
                    noise = np.random.normal(0, self.config.gaussian_noise_std)
                    effect.intensity = np.clip(effect.intensity + noise, 0.0, 1.0)
                
                if hasattr(effect, 'weight'):
                    # Add Gaussian noise to weight
                    noise = np.random.normal(0, self.config.gaussian_noise_std)
                    effect.weight = np.clip(effect.weight + noise, 0.0, 1.0)
                
                if hasattr(effect, 'parameters'):
                    # Mutate effect parameters
                    self._mutate_effect_parameters(effect)
    
    def _mutate_effect_parameters(self, effect):
        """Mutate parameters of a specific effect"""
        if not hasattr(effect, 'parameters') or not effect.parameters:
            return
        
        for param_name, param_value in effect.parameters.items():
            if isinstance(param_value, (int, float)):
                # Add noise to numeric parameters
                if isinstance(param_value, int):
                    noise = int(np.random.normal(0, self.config.mutation_strength * abs(param_value)))
                    effect.parameters[param_name] = max(0, param_value + noise)
                else:
                    noise = np.random.normal(0, self.config.mutation_strength * abs(param_value))
                    effect.parameters[param_name] = param_value + noise
            elif isinstance(param_value, bool):
                # Flip boolean parameters with small probability
                if random.random() < self.config.mutation_rate:
                    effect.parameters[param_name] = not param_value
            elif isinstance(param_value, str):
                # Mutate string parameters (e.g., mode names)
                if random.random() < self.config.mutation_rate:
                    effect.parameters[param_name] = self._mutate_string_parameter(param_value)
    
    def _mutate_string_parameter(self, value: str) -> str:
        """Mutate string parameters"""
        # Simple string mutation strategies
        if value.lower() in ['low', 'medium', 'high']:
            options = ['low', 'medium', 'high']
            return random.choice([opt for opt in options if opt != value.lower()])
        elif value.lower() in ['linear', 'exponential', 'logarithmic']:
            options = ['linear', 'exponential', 'logarithmic']
            return random.choice([opt for opt in options if opt != value.lower()])
        else:
            # Default: add random character
            if len(value) > 0:
                pos = random.randint(0, len(value))
                char = random.choice('abcdefghijklmnopqrstuvwxyz')
                return value[:pos] + char + value[pos:]
            return value
    
    def _add_random_effect(self, pack: Pack):
        """Add a random effect to the pack"""
        try:
            # Get all available effects directly from the effect registry
            from ..effects import EffectRegistry
            effect_registry = EffectRegistry()
            available_effects = effect_registry.list_effects()
            
            # Filter out effects already in the pack
            existing_effects = {effect.effect for effect in pack.effects} if hasattr(pack, 'effects') and pack.effects else set()
            available_effects = [eff for eff in available_effects if eff not in existing_effects]
            
            if not available_effects:
                return  # No new effects to add
            
            # Choose a random effect
            effect_name = random.choice(available_effects)
            
            # Create effect configuration with random parameters
            from ..pack_system import EffectConfig
            effect_config = EffectConfig(
                effect=effect_name,
                weight=random.uniform(0.1, 0.9),
                direction=random.choice(['up', 'down', 'neutral']),
                parameters=self._generate_random_parameters(effect_name)
            )
            
            pack.effects.append(effect_config)
            logger.debug(f"Added random effect: {effect_name}")
            
        except Exception as e:
            logger.warning(f"Failed to add random effect: {e}")
    
    def _remove_random_effect(self, pack: Pack):
        """Remove a random effect from the pack"""
        if not hasattr(pack, 'effects') or not pack.effects:
            return
        
        # Choose a random effect to remove
        effect_to_remove = random.choice(pack.effects)
        pack.effects.remove(effect_to_remove)
        logger.debug(f"Removed effect: {effect_to_remove.effect}")
    
    def _generate_random_parameters(self, effect_name: str) -> Dict[str, Any]:
        """Generate random parameters for a given effect"""
        # Common parameter ranges for different effect types
        parameter_templates = {
            # Sampler effects
            'temperature': {'temperature': random.uniform(0.1, 2.0)},
            'top_p': {'top_p': random.uniform(0.1, 1.0)},
            'frequency_penalty': {'frequency_penalty': random.uniform(0.0, 2.0)},
            'presence_penalty': {'presence_penalty': random.uniform(0.0, 2.0)},
            
            # Attention effects
            'attention_focus': {
                'focus_strength': random.uniform(0.1, 1.0),
                'focus_layers': random.sample(range(1, 13), random.randint(1, 3))
            },
            'attention_masking': {
                'mask_probability': random.uniform(0.1, 0.5),
                'mask_layers': random.sample(range(1, 13), random.randint(1, 3))
            },
            
            # Steering effects
            'steering': {
                'steering_vector': [random.uniform(-1, 1) for _ in range(768)],
                'steering_strength': random.uniform(0.1, 1.0)
            },
            
            # Memory effects
            'kv_decay': {
                'decay_factor': random.uniform(0.8, 1.0),
                'decay_layers': random.sample(range(1, 13), random.randint(1, 3))
            },
            'kv_compression': {
                'compression_ratio': random.uniform(0.1, 0.9),
                'compression_layers': random.sample(range(1, 13), random.randint(1, 3))
            },
            
            # Default parameters for unknown effects
            'default': {
                'strength': random.uniform(0.1, 1.0),
                'enabled': True,
                'mode': random.choice(['low', 'medium', 'high'])
            }
        }
        
        # Return specific parameters if available, otherwise default
        return parameter_templates.get(effect_name, parameter_templates['default'])
    
    def _mutate_weights(self, pack: Pack):
        """Mutate pack-level weights"""
        if not hasattr(pack, 'weights') or not pack.weights:
            return
        
        for weight_name, weight_value in pack.weights.items():
            if isinstance(weight_value, (int, float)):
                noise = np.random.normal(0, self.config.gaussian_noise_std)
                pack.weights[weight_name] = np.clip(weight_value + noise, 0.0, 1.0)
    
    def _mutate_parameters(self, pack: Pack):
        """Mutate pack-level parameters"""
        if not hasattr(pack, 'parameters') or not pack.parameters:
            return
        
        for param_name, param_value in pack.parameters.items():
            if isinstance(param_value, (int, float)):
                noise = np.random.normal(0, self.config.mutation_strength * abs(param_value))
                if isinstance(param_value, int):
                    pack.parameters[param_name] = max(0, int(param_value + noise))
                else:
                    pack.parameters[param_name] = param_value + noise
            elif isinstance(param_value, bool):
                if random.random() < self.config.mutation_rate:
                    pack.parameters[param_name] = not param_value
    
    def _mutate_metadata(self, pack: Pack):
        """Mutate pack metadata"""
        if not hasattr(pack, 'metadata') or not pack.metadata:
            return
        
        # Mutate numeric metadata
        for key, value in pack.metadata.items():
            if isinstance(value, (int, float)):
                if random.random() < self.config.mutation_rate:
                    noise = np.random.normal(0, self.config.mutation_strength * abs(value))
                    pack.metadata[key] = value + noise

class PackCrossover:
    """Handles crossover operations for pack parameters"""
    
    def __init__(self, config: CrossoverConfig = None):
        self.config = config or CrossoverConfig()
    
    def crossover(self, parent1: Pack, parent2: Pack) -> Tuple[Pack, Pack]:
        """
        Perform crossover between two parent packs.
        
        Args:
            parent1: First parent pack
            parent2: Second parent pack
            
        Returns:
            Tuple of (child1, child2)
        """
        if random.random() > self.config.crossover_rate:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Create children
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
        # Perform crossover based on type
        if self.config.crossover_type == 'uniform':
            self._uniform_crossover(child1, child2, parent1, parent2)
        elif self.config.crossover_type == 'single_point':
            self._single_point_crossover(child1, child2, parent1, parent2)
        elif self.config.crossover_type == 'two_point':
            self._two_point_crossover(child1, child2, parent1, parent2)
        elif self.config.crossover_type == 'arithmetic':
            self._arithmetic_crossover(child1, child2, parent1, parent2)
        else:
            # Default to uniform crossover
            self._uniform_crossover(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _uniform_crossover(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Uniform crossover: randomly swap features between parents"""
        # Crossover effects
        if hasattr(parent1, 'effects') and hasattr(parent2, 'effects'):
            self._crossover_effects(child1, child2, parent1, parent2)
        
        # Crossover weights
        if hasattr(parent1, 'weights') and hasattr(parent2, 'weights'):
            self._crossover_weights(child1, child2, parent1, parent2)
        
        # Crossover parameters
        if hasattr(parent1, 'parameters') and hasattr(parent2, 'parameters'):
            self._crossover_parameters(child1, child2, parent1, parent2)
    
    def _single_point_crossover(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Single-point crossover: split at one point"""
        # For packs, we'll use a simplified single-point crossover
        # by randomly selecting which parent's features to use
        if random.random() < 0.5:
            # Child1 gets parent1's effects, child2 gets parent2's effects
            if hasattr(parent1, 'effects'):
                child1.effects = copy.deepcopy(parent1.effects)
            if hasattr(parent2, 'effects'):
                child2.effects = copy.deepcopy(parent2.effects)
        else:
            # Child1 gets parent2's effects, child2 gets parent1's effects
            if hasattr(parent2, 'effects'):
                child1.effects = copy.deepcopy(parent2.effects)
            if hasattr(parent1, 'effects'):
                child2.effects = copy.deepcopy(parent1.effects)
    
    def _two_point_crossover(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Two-point crossover: split at two points"""
        # Similar to single-point but with two crossover points
        # For simplicity, we'll use uniform crossover with two random decisions
        self._uniform_crossover(child1, child2, parent1, parent2)
    
    def _arithmetic_crossover(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Arithmetic crossover: blend numeric values"""
        alpha = self.config.blend_alpha
        
        # Blend effects
        if hasattr(parent1, 'effects') and hasattr(parent2, 'effects'):
            self._blend_effects(child1, child2, parent1, parent2, alpha)
        
        # Blend weights
        if hasattr(parent1, 'weights') and hasattr(parent2, 'weights'):
            self._blend_weights(child1, child2, parent1, parent2, alpha)
        
        # Blend parameters
        if hasattr(parent1, 'parameters') and hasattr(parent2, 'parameters'):
            self._blend_parameters(child1, child2, parent1, parent2, alpha)
    
    def _crossover_effects(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Crossover effects between parents - can mix effects from both parents"""
        if not hasattr(parent1, 'effects') or not hasattr(parent2, 'effects'):
            return
        
        # Collect all unique effects from both parents
        all_effects = {}
        
        # Add effects from parent1
        if parent1.effects:
            for effect in parent1.effects:
                all_effects[effect.effect] = effect
        
        # Add effects from parent2 (may override parent1's version)
        if parent2.effects:
            for effect in parent2.effects:
                all_effects[effect.effect] = effect
        
        # Create children by randomly selecting effects
        child1.effects = []
        child2.effects = []
        
        for effect_name, effect in all_effects.items():
            # Each child gets a copy of the effect with some probability
            if random.random() < 0.7:  # 70% chance to inherit each effect
                child1.effects.append(copy.deepcopy(effect))
            
            if random.random() < 0.7:  # 70% chance to inherit each effect
                child2.effects.append(copy.deepcopy(effect))
        
        # Ensure children have at least one effect
        if not child1.effects and all_effects:
            effect_name = random.choice(list(all_effects.keys()))
            child1.effects.append(copy.deepcopy(all_effects[effect_name]))
        
        if not child2.effects and all_effects:
            effect_name = random.choice(list(all_effects.keys()))
            child2.effects.append(copy.deepcopy(all_effects[effect_name]))
    
    def _crossover_weights(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Crossover weights between parents"""
        if not parent1.weights or not parent2.weights:
            return
        
        for key in parent1.weights:
            if key in parent2.weights:
                if random.random() < 0.5:
                    child1.weights[key] = parent1.weights[key]
                    child2.weights[key] = parent2.weights[key]
                else:
                    child1.weights[key] = parent2.weights[key]
                    child2.weights[key] = parent1.weights[key]
    
    def _crossover_parameters(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack):
        """Crossover parameters between parents"""
        if not parent1.parameters or not parent2.parameters:
            return
        
        for key in parent1.parameters:
            if key in parent2.parameters:
                if random.random() < 0.5:
                    child1.parameters[key] = parent1.parameters[key]
                    child2.parameters[key] = parent2.parameters[key]
                else:
                    child1.parameters[key] = parent2.parameters[key]
                    child2.parameters[key] = parent1.parameters[key]
    
    def _blend_effects(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack, alpha: float):
        """Blend effects using arithmetic crossover"""
        if not parent1.effects or not parent2.effects:
            return
        
        for i in range(min(len(parent1.effects), len(parent2.effects))):
            effect1 = parent1.effects[i]
            effect2 = parent2.effects[i]
            
            # Blend numeric attributes
            if hasattr(effect1, 'intensity') and hasattr(effect2, 'intensity'):
                child1.effects[i].intensity = alpha * effect1.intensity + (1 - alpha) * effect2.intensity
                child2.effects[i].intensity = alpha * effect2.intensity + (1 - alpha) * effect1.intensity
            
            if hasattr(effect1, 'weight') and hasattr(effect2, 'weight'):
                child1.effects[i].weight = alpha * effect1.weight + (1 - alpha) * effect2.weight
                child2.effects[i].weight = alpha * effect2.weight + (1 - alpha) * effect1.weight
    
    def _blend_weights(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack, alpha: float):
        """Blend weights using arithmetic crossover"""
        if not parent1.weights or not parent2.weights:
            return
        
        for key in parent1.weights:
            if key in parent2.weights and isinstance(parent1.weights[key], (int, float)):
                child1.weights[key] = alpha * parent1.weights[key] + (1 - alpha) * parent2.weights[key]
                child2.weights[key] = alpha * parent2.weights[key] + (1 - alpha) * parent1.weights[key]
    
    def _blend_parameters(self, child1: Pack, child2: Pack, parent1: Pack, parent2: Pack, alpha: float):
        """Blend parameters using arithmetic crossover"""
        if not parent1.parameters or not parent2.parameters:
            return
        
        for key in parent1.parameters:
            if key in parent2.parameters and isinstance(parent1.parameters[key], (int, float)):
                child1.parameters[key] = alpha * parent1.parameters[key] + (1 - alpha) * parent2.parameters[key]
                child2.parameters[key] = alpha * parent2.parameters[key] + (1 - alpha) * parent1.parameters[key]

# Convenience functions
def mutate_pack(pack: Pack, generation: int = 0, config: MutationConfig = None) -> Pack:
    """Convenience function to mutate a pack"""
    mutator = PackMutator(config)
    return mutator.mutate(pack, generation)

def crossover_packs(parent1: Pack, parent2: Pack, config: CrossoverConfig = None) -> Tuple[Pack, Pack]:
    """Convenience function to crossover two packs"""
    crossover = PackCrossover(config)
    return crossover.crossover(parent1, parent2)
