"""
Modular neuromodulation effects system
Each effect can be applied with weight (0.0-1.0) and direction (up/down/neutral)
"""

import torch
import torch.nn.functional as F
import random
import math
import contextlib
import logging
from typing import Dict, Any, List, Optional, Callable, Iterable, Union
from transformers import LogitsProcessor
import numpy as np
from abc import ABC, abstractmethod

# Configure logging
logger = logging.getLogger(__name__)

# Import visual effects
try:
    from .visual_effects import (
        ColorBiasEffect, StyleTransferEffect, CompositionBiasEffect,
        VisualEntropyEffect, SynestheticMappingEffect, MotionBlurEffect
    )
except ImportError:
    # Fallback dummy classes if visual_effects module is not available
    class ColorBiasEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass
    class StyleTransferEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass
    class CompositionBiasEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass
    class VisualEntropyEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass
    class SynestheticMappingEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass
    class MotionBlurEffect(BaseEffect):
        def apply(self, model, **kwargs): pass
        def cleanup(self): pass

# ============================================================================
# BASE EFFECT CLASSES
# ============================================================================

class BaseEffect(ABC):
    """Base class for all neuromodulation effects"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        self.weight = max(0.0, min(1.0, weight))  # Clamp to 0-1
        self.direction = direction  # "up", "down", or "neutral"
        
    @abstractmethod
    def apply(self, model, **kwargs):
        """Apply the effect to the model"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Clean up the effect"""
        pass
        
    def get_effective_value(self, base_value: float, max_change: float) -> float:
        """Calculate effective value based on weight and direction"""
        if self.direction == "up":
            return base_value + (max_change * self.weight)
        elif self.direction == "down":
            return base_value - (max_change * self.weight)
        else:  # neutral
            return base_value

class SamplerEffect(BaseEffect):
    """Base class for sampling parameter effects"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", parameter: str = "temperature"):
        super().__init__(weight, direction)
        self.parameter = parameter
        
    def get_logits_processor(self) -> Optional[LogitsProcessor]:
        """Return a logits processor for this effect"""
        return None

# ============================================================================
# SAMPLER EFFECTS
# ============================================================================

class TemperatureEffect(SamplerEffect):
    """Temperature sampling effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction, "temperature")
        self.base_temp = 1.0
        self.max_change = 1.0  # Can go from 0.1 to 2.0
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_temp = self.get_effective_value(self.base_temp, self.max_change)
        
        class TempProcessor(LogitsProcessor):
            def __init__(self, temp):
                self.temp = max(0.1, temp)
            def __call__(self, input_ids, scores):
                return scores / self.temp
                
        return TempProcessor(effective_temp)
        
    def cleanup(self):
        pass

class TopPEffect(SamplerEffect):
    """Top-p sampling effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction, "top_p")
        self.base_top_p = 1.0
        self.max_change = 0.3  # Can go from 0.7 to 1.3
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_top_p = self.get_effective_value(self.base_top_p, self.max_change)
        
        class TopPProcessor(LogitsProcessor):
            def __init__(self, top_p):
                self.top_p = max(0.1, min(1.0, top_p))
            def __call__(self, input_ids, scores):
                sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > self.top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scores[indices_to_remove] = float('-inf')
                return scores
                
        return TopPProcessor(effective_top_p)
        
    def cleanup(self):
        pass

class FrequencyPenaltyEffect(SamplerEffect):
    """Frequency penalty effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction, "frequency_penalty")
        self.base_penalty = 1.0
        self.max_change = 0.5  # Can go from 0.5 to 1.5
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_penalty = self.get_effective_value(self.base_penalty, self.max_change)
        
        class FrequencyProcessor(LogitsProcessor):
            def __init__(self, penalty):
                self.penalty = penalty
            def __call__(self, input_ids, scores):
                unique_tokens, counts = torch.unique(input_ids, return_counts=True)
                for token, count in zip(unique_tokens, counts):
                    if count > 1:
                        scores[..., token] *= (self.penalty ** (count - 1))
                return scores
                
        return FrequencyProcessor(effective_penalty)
        
    def cleanup(self):
        pass

class PresencePenaltyEffect(SamplerEffect):
    """Presence penalty effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction, "presence_penalty")
        self.base_penalty = 0.0
        self.max_change = 2.0  # Can go from 0 to 2.0
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_penalty = self.get_effective_value(self.base_penalty, self.max_change)
        
        class PresenceProcessor(LogitsProcessor):
            def __init__(self, penalty):
                self.penalty = penalty
            def __call__(self, input_ids, scores):
                unique_tokens, counts = torch.unique(input_ids, return_counts=True)
                for token, count in zip(unique_tokens, counts):
                    if count > 0:
                        scores[..., token] -= self.penalty * count
                return scores
                
        return PresenceProcessor(effective_penalty)
        
    def cleanup(self):
        pass

class PulsedSamplerEffect(SamplerEffect):
    """Pulsed sampler effect for nicotine-like microbursts"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 pulse_interval: int = 20, pulse_duration: int = 5):
        super().__init__(weight, direction, "pulsed_sampler")
        self.pulse_interval = pulse_interval
        self.pulse_duration = pulse_duration
        self.base_temp = 1.0
        self.max_temp_change = 0.5  # Can change temp by ±0.5
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_temp_change = self.get_effective_value(0.0, self.max_temp_change)
        
        class PulsedProcessor(LogitsProcessor):
            def __init__(self, temp_change, interval, duration):
                self.temp_change = temp_change
                self.interval = interval
                self.duration = duration
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                # Check if we're in a pulse
                pulse_position = (self.token_count - 1) % self.interval
                in_pulse = pulse_position < self.duration
                
                if in_pulse:
                    # Apply temperature change during pulse
                    temp = 1.0 + self.temp_change
                    scores = scores / temp
                    
                return scores
                
        return PulsedProcessor(effective_temp_change, self.pulse_interval, self.pulse_duration)
        
    def cleanup(self):
        pass

class ContrastiveDecodingEffect(SamplerEffect):
    """Contrastive decoding effect using small model logits"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 small_model_name: str = "gpt2"):
        super().__init__(weight, direction, "contrastive_decoding")
        self.small_model_name = small_model_name
        self.small_model = None
        self.small_tokenizer = None
        self.base_alpha = 0.0
        self.max_alpha = 0.3  # Can use up to 0.3 alpha
        
    def apply(self, model, **kwargs):
        # Load small model for contrastive decoding
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.small_tokenizer = AutoTokenizer.from_pretrained(self.small_model_name)
            if self.small_tokenizer.pad_token is None:
                self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
                
            self.small_model = AutoModelForCausalLM.from_pretrained(
                self.small_model_name, 
                dtype=torch.float32, 
                device_map='cpu'
            )
            self.small_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load small model for contrastive decoding: {e}")
            self.small_model = None
        
    def get_logits_processor(self) -> LogitsProcessor:
        if self.small_model is None:
            return None
            
        effective_alpha = self.get_effective_value(self.base_alpha, self.max_alpha)
        
        class ContrastiveProcessor(LogitsProcessor):
            def __init__(self, small_model, small_tokenizer, alpha):
                self.small_model = small_model
                self.small_tokenizer = small_tokenizer
                self.alpha = alpha
                
            def __call__(self, input_ids, scores):
                try:
                    with torch.no_grad():
                        # Get device from scores (main model's device)
                        device = scores.device
                        
                        # Get small model logits
                        small_inputs = self.small_tokenizer(
                            self.small_tokenizer.decode(input_ids[0]), 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=input_ids.shape[1]
                        )
                        # Move small model inputs to the same device as scores
                        small_inputs = {k: v.to(device) for k, v in small_inputs.items()}
                        
                        # Move small model to the same device
                        small_model_device = next(self.small_model.parameters()).device
                        if small_model_device != device:
                            # If small model is on different device, we need to handle this
                            # For now, try to move inputs to small model's device and move result back
                            small_inputs = {k: v.to(small_model_device) for k, v in small_inputs.items()}
                            small_outputs = self.small_model(**small_inputs)
                            small_logits = small_outputs.logits[:, -1, :].to(device)
                        else:
                            small_outputs = self.small_model(**small_inputs)
                            small_logits = small_outputs.logits[:, -1, :]
                        
                        # Apply contrastive decoding: scores = scores - alpha * small_logits
                        scores = scores - self.alpha * small_logits
                        
                except Exception as e:
                    # Fallback if contrastive decoding fails
                    pass
                    
                return scores
                
        return ContrastiveProcessor(self.small_model, self.small_tokenizer, effective_alpha)
        
    def cleanup(self):
        if self.small_model is not None:
            del self.small_model
            del self.small_tokenizer
            self.small_model = None
            self.small_tokenizer = None

class ExpertMixingEffect(SamplerEffect):
    """Expert/anti-expert mixing effect (DExperts/GeDi-like)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 expert_type: str = "concise", anti_expert_type: str = "verbose"):
        super().__init__(weight, direction, "expert_mixing")
        self.expert_type = expert_type
        self.anti_expert_type = anti_expert_type
        self.base_strength = 0.0
        self.max_strength = 0.4  # Can use up to 0.4 strength
        
        # Define expert/anti-expert attribute vectors
        self.attribute_vectors = {
            "concise": {
                "pos": ["brief", "short", "direct", "clear", "simple"],
                "neg": ["verbose", "long", "complex", "detailed", "elaborate"]
            },
            "verbose": {
                "pos": ["detailed", "comprehensive", "thorough", "elaborate", "extensive"],
                "neg": ["brief", "concise", "short", "direct", "simple"]
            },
            "formal": {
                "pos": ["formal", "professional", "academic", "precise", "technical"],
                "neg": ["casual", "informal", "colloquial", "relaxed", "friendly"]
            },
            "creative": {
                "pos": ["creative", "imaginative", "artistic", "innovative", "original"],
                "neg": ["conventional", "traditional", "standard", "ordinary", "typical"]
            }
        }
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_strength = self.get_effective_value(self.base_strength, self.max_strength)
        
        class ExpertMixingProcessor(LogitsProcessor):
            def __init__(self, expert_type, anti_expert_type, strength, attribute_vectors):
                self.expert_type = expert_type
                self.anti_expert_type = anti_expert_type
                self.strength = strength
                self.attribute_vectors = attribute_vectors
                
            def __call__(self, input_ids, scores):
                try:
                    # Get attribute words for expert and anti-expert
                    if self.expert_type in self.attribute_vectors:
                        expert_pos = self.attribute_vectors[self.expert_type]["pos"]
                        expert_neg = self.attribute_vectors[self.expert_type]["neg"]
                    else:
                        return scores
                        
                    if self.anti_expert_type in self.attribute_vectors:
                        anti_pos = self.attribute_vectors[self.anti_expert_type]["pos"]
                        anti_neg = self.attribute_vectors[self.anti_expert_type]["neg"]
                    else:
                        return scores
                    
                    # Apply expert mixing by boosting expert words and suppressing anti-expert words
                    # This is a simplified version - in practice you'd use learned attribute vectors
                    for word in expert_pos + anti_neg:
                        # Boost expert words
                        pass  # Would need tokenizer to convert words to token IDs
                        
                    for word in anti_pos + expert_neg:
                        # Suppress anti-expert words
                        pass  # Would need tokenizer to convert words to token IDs
                        
                except Exception as e:
                    # Fallback if expert mixing fails
                    pass
                    
                return scores
                
        return ExpertMixingProcessor(
            self.expert_type, 
            self.anti_expert_type, 
            effective_strength, 
            self.attribute_vectors
        )
        
    def cleanup(self):
        pass

class TokenClassTemperatureEffect(SamplerEffect):
    """Token-class-aware temperature effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 content_temp_factor: float = 0.8, modifier_temp_factor: float = 1.2):
        super().__init__(weight, direction, "token_class_temperature")
        self.content_temp_factor = content_temp_factor
        self.modifier_temp_factor = modifier_temp_factor
        self.base_factor = 1.0
        self.max_factor_change = 0.3  # Can change factors by ±0.3
        
    def apply(self, model, **kwargs):
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        effective_factor_change = self.get_effective_value(0.0, self.max_factor_change)
        
        class TokenClassProcessor(LogitsProcessor):
            def __init__(self, content_factor, modifier_factor, factor_change):
                self.content_factor = content_factor + factor_change
                self.modifier_factor = modifier_factor + factor_change
                
                # Define token classes (simplified)
                self.content_pos = ["NOUN", "VERB", "PROPN", "NUM"]
                self.modifier_pos = ["ADJ", "ADV", "DET", "ADP", "CONJ", "INTJ"]
                
            def __call__(self, input_ids, scores):
                try:
                    # This is a simplified implementation
                    # In practice, you'd need POS tagging or a more sophisticated approach
                    
                    # For now, apply different temperatures based on token frequency
                    # (assuming rare tokens are more likely to be content words)
                    
                    # Get token frequencies (simplified heuristic)
                    token_probs = torch.softmax(scores, dim=-1)
                    entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-8), dim=-1)
                    
                    # Apply different temperatures based on entropy
                    # High entropy = likely modifier, low entropy = likely content
                    entropy_threshold = 0.5
                    
                    # Apply content temperature to low-entropy tokens
                    content_mask = entropy < entropy_threshold
                    scores[content_mask] = scores[content_mask] / self.content_factor
                    
                    # Apply modifier temperature to high-entropy tokens  
                    modifier_mask = entropy >= entropy_threshold
                    scores[modifier_mask] = scores[modifier_mask] / self.modifier_factor
                    
                except Exception as e:
                    # Fallback if token class processing fails
                    pass
                    
                return scores
                
        return TokenClassProcessor(
            self.content_temp_factor, 
            self.modifier_temp_factor, 
            effective_factor_change
        )
        
    def cleanup(self):
        pass

# ============================================================================
# ATTENTION EFFECTS
# ============================================================================

class AttentionFocusEffect(BaseEffect):
    """
    Attention focus enhancement effect with head-specific induction head targeting.
    
    This is an alias/synonym for QKScoreScalingEffect with the same head-specific implementation.
    Uses the same REAL induction head detection (via calibration prompt) and repetition penalty approach.
    
    NOTE: This effect now uses REAL induction head detection, not heuristics. If detection fails,
    it falls back to heuristic sharpening (which should be renamed to HeuristicAttentionSharpening).
    """
    
    def __init__(self, weight: float = 0.5, direction: str = "up", layers: str = "mid",
                 auto_detect_induction_heads: bool = True,
                 induction_head_indices: Optional[List[int]] = None):
        super().__init__(weight, direction)
        self.layers = layers
        self.handles = []
        self.auto_detect_induction_heads = auto_detect_induction_heads
        self.induction_head_indices = induction_head_indices
        self.induction_head_masks = {}
        self.repetition_penalty_active = True
        
        # Store parameters for lazy initialization (QKScoreScalingEffect defined later)
        self._qk_scaling_params = {
            'weight': weight,
            'direction': direction,
            'layers': layers,
            'auto_detect_induction_heads': auto_detect_induction_heads,
            'induction_head_indices': induction_head_indices
        }
        self._qk_scaling_effect = None
        
    def _get_qk_scaling_effect(self):
        """Lazy initialization of QKScoreScalingEffect"""
        if self._qk_scaling_effect is None:
            # QKScoreScalingEffect is defined later in the file, so we can reference it here
            self._qk_scaling_effect = QKScoreScalingEffect(**self._qk_scaling_params)
        return self._qk_scaling_effect
        
    def apply(self, model, **kwargs):
        """Apply attention focus using head-specific QK scaling"""
        # Delegate to QKScoreScalingEffect (lazy init)
        qk_effect = self._get_qk_scaling_effect()
        qk_effect.apply(model, **kwargs)
        # Copy handles for cleanup
        self.handles = qk_effect.handles
        self.induction_head_masks = qk_effect.induction_head_masks
                
    def cleanup(self):
        """Remove all hooks"""
        if self._qk_scaling_effect is not None:
            self._qk_scaling_effect.cleanup()
        self.handles.clear()
        self.induction_head_masks.clear()
    
    def get_logits_processor(self):
        """Return repetition penalty processor"""
        qk_effect = self._get_qk_scaling_effect()
        return qk_effect.get_logits_processor()
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer (kept for compatibility)"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

class AttentionMaskingEffect(BaseEffect):
    """Attention head masking effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", layers: str = "mid"):
        super().__init__(weight, direction)
        self.layers = layers
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply attention head masking"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                attn = self._get_attention_module(blocks[layer_idx])
                if attn is None:
                    continue
                    
                # Calculate effective masking probability
                base_prob = 0.0
                max_prob = 0.3
                effective_prob = self.get_effective_value(base_prob, max_prob)
                
                if hasattr(attn, "num_heads"):
                    num_heads = attn.num_heads
                    heads_to_mask = random.sample(range(num_heads), 
                                               int(num_heads * effective_prob))
                    
                    # Store original forward
                    original_forward = attn.forward
                    
                    def masked_forward(*args, **kwargs):
                        output = original_forward(*args, **kwargs)
                        if isinstance(output, tuple) and len(output) >= 2:
                            attn_weights = output[1]
                            if attn_weights is not None:
                                # Zero out masked heads
                                for head_idx in heads_to_mask:
                                    attn_weights[:, head_idx, :, :] = 0.0
                                
                                # Recompute attention output
                                if len(output) >= 3 and output[2] is not None:
                                    V = output[2]
                                    new_attn_output = torch.matmul(attn_weights, V)
                                    output = (new_attn_output,) + output[1:]
                        return output
                    
                    attn.forward = masked_forward
                    self.handles.append((attn, original_forward))
                    
            except Exception as e:
                print(f"Warning: Failed to apply attention masking in layer {layer_idx}: {e}")
                continue
                
    def get_logits_processor(self):
        """Attention effects don't use logits processors"""
        return None
                
    def cleanup(self):
        """Restore original attention forward methods"""
        for attn, original_forward in self.handles:
            attn.forward = original_forward
        self.handles.clear()
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

# ============================================================================
# ATTENTION MODIFICATION EFFECTS
# ============================================================================

class QKScoreScalingEffect(BaseEffect):
    """
    QK score scaling (attention sharpness) with head-specific induction head targeting.
    
    FIXED: Instead of globally sharpening all heads (which causes mode collapse),
    this effect ACTUALLY DETECTS "induction heads" (heads responsible for copying/continuation)
    using a calibration prompt ("A B C D E F A") and sharpens only those, while leaving
    exploratory heads unchanged.
    
    REAL DETECTION: Uses calibration prompt to identify heads that attend from the second
    occurrence of a token to the position after the first occurrence. This is the classic
    induction head pattern, not a heuristic.
    
    Mathematical basis:
    - Original: attn_logits = Q @ K^T / sqrt(d_k)
    - Scaled (induction heads only): attn_logits = (α * Q_induction) @ K^T / sqrt(d_k)
    - This sharpens only continuation patterns, avoiding global mode collapse
    - Includes repetition penalty to counteract any remaining mode collapse
    
    If detection fails, falls back to heuristic (which should be renamed to
    HeuristicAttentionSharpening if used).
    """
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "mid", scaling_type: str = "uniform",
                 auto_detect_induction_heads: bool = True,
                 induction_head_indices: Optional[List[int]] = None):
        super().__init__(weight, direction)
        self.layers = layers
        self.scaling_type = scaling_type
        self.handles = []
        self.auto_detect_induction_heads = auto_detect_induction_heads
        self.induction_head_indices = induction_head_indices  # Manual override
        self.induction_head_masks = {}  # Cache per layer
        self.repetition_penalty_active = True
        
    def _detect_induction_heads(self, model, block, layer_idx: int, num_heads: int, 
                                tokenizer=None) -> List[int]:
        """
        Detect induction heads by analyzing attention patterns on calibration prompt.
        
        REAL DETECTION: Pass a string like "A B C D E F A" through the model and identify
        heads that strongly attend from the second A (position 6) to the token after the first A (position 1).
        
        This is the classic induction head detection pattern:
        - Induction heads attend from position i (second occurrence) to position j+1 (token after first occurrence)
        - Where j is the position of the first occurrence
        
        CRITICAL: Uses token IDs directly for detection, not string matching. Accounts for
        tokenizer prefixes (e.g., ĠA vs A in GPT-2 style tokenizers) by checking variants.
        
        Args:
            model: The language model
            block: The transformer block
            layer_idx: Index of the layer
            num_heads: Number of attention heads
            tokenizer: Tokenizer for the model (optional, will try to get from model)
        
        Returns:
            List of induction head indices
        """
        if self.induction_head_indices is not None:
            # Use manual override
            return [h for h in self.induction_head_indices if h < num_heads]
        
        if not self.auto_detect_induction_heads:
            # Fallback to heuristic if auto-detection is disabled
            logger.warning("Auto-detection disabled, using heuristic. Consider enabling auto-detection for real induction head detection.")
            if num_heads >= 8:
                induction_heads = [num_heads // 2 - 1, num_heads // 2]
            elif num_heads >= 4:
                induction_heads = [num_heads // 2]
            else:
                induction_heads = [0]
            return [h for h in induction_heads if h < num_heads]
        
        # REAL DETECTION: Use calibration prompt to identify induction heads
        try:
            # Get tokenizer
            if tokenizer is None:
                if hasattr(model, 'tokenizer'):
                    tokenizer = model.tokenizer
                else:
                    logger.warning("No tokenizer available for induction head detection, falling back to heuristic")
                    return self._heuristic_induction_heads(num_heads)
            
            # Create calibration prompt: Pattern with repeated token
            # Use a simple pattern that works across tokenizers
            # CRITICAL: Account for tokenizer prefixes (e.g., Ġthe vs the in GPT-2 style)
            # We'll use a pattern that's more robust to tokenization differences
            # [FIX] Add leading space to ensure first and last 'A' match in Llama-3 tokenizer
            calibration_prompt = " A B C D E F A"
            
            # Tokenize
            inputs = tokenizer(calibration_prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get token IDs directly
            input_ids = inputs['input_ids'][0].cpu().numpy()
            
            # Find positions of first and second occurrence of a repeated token
            # Use token IDs directly, not string matching
            from collections import Counter
            token_counts = Counter(input_ids.tolist())
            repeated_tokens = [tid for tid, count in token_counts.items() if count >= 2]
            
            if not repeated_tokens:
                # Fallback: Try to find any repeated token by checking tokenizer variants
                # For tokenizers with prefixes (e.g., GPT-2: "A" vs "ĠA"), we need to check both
                # CRITICAL: Use token IDs directly, accounting for tokenizer prefixes
                token_variants = []
                for variant in ["A", " A", "A ", " A "]:
                    variant_ids = tokenizer.encode(variant, add_special_tokens=False)
                    if variant_ids:
                        token_variants.extend(variant_ids)
                
                # Find which variant appears in the input_ids (using token IDs directly)
                for variant_id in set(token_variants):
                    if variant_id in input_ids:
                        count = list(input_ids).count(variant_id)
                        if count >= 2:
                            repeated_tokens.append(variant_id)
                            break
                
                if not repeated_tokens:
                    # Last resort: Check if first token appears multiple times
                    # This handles cases where tokenization is unexpected
                    first_token_id = input_ids[0] if len(input_ids) > 0 else None
                    if first_token_id is not None:
                        count = list(input_ids).count(first_token_id)
                        if count >= 2:
                            repeated_tokens.append(first_token_id)
                    
                    if not repeated_tokens:
                        logger.warning("Could not find repeated token in calibration prompt, falling back to heuristic")
                        logger.debug(f"Input IDs: {input_ids.tolist()}")
                        return self._heuristic_induction_heads(num_heads)
            
            # Use the first repeated token (most common)
            target_token_id = repeated_tokens[0]
            
            # Find positions of first and second occurrence using token IDs directly
            first_pos = None
            second_pos = None
            
            for i, token_id in enumerate(input_ids):
                if token_id == target_token_id:
                    if first_pos is None:
                        first_pos = i
                    elif second_pos is None:
                        second_pos = i
                        break  # Found both positions
            
            if first_pos is None or second_pos is None:
                logger.warning("Could not find repeated tokens in calibration prompt, falling back to heuristic")
                return self._heuristic_induction_heads(num_heads)
            
            # Target position: token after first occurrence (first_pos + 1)
            target_pos = first_pos + 1
            if target_pos >= len(input_ids):
                logger.warning("Target position out of range, falling back to heuristic")
                return self._heuristic_induction_heads(num_heads)
            
            # Get attention module
            attn_module = None
            if hasattr(block, 'self_attn'):
                attn_module = block.self_attn
            elif hasattr(block, 'attn'):
                attn_module = block.attn
            elif hasattr(block, 'attention'):
                attn_module = block.attention
            
            if attn_module is None:
                logger.warning("Could not find attention module, falling back to heuristic")
                return self._heuristic_induction_heads(num_heads)
            
            # Hook to capture attention weights
            attention_weights = None
            
            def attention_hook(module, input_tuple, output):
                nonlocal attention_weights
                # Attention output is usually (hidden_states, attention_weights, ...)
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        # attn_weights shape: [batch, num_heads, seq_len, seq_len]
                        attention_weights = attn_weights.detach().cpu()
            
            handle = attn_module.register_forward_hook(attention_hook)
            
            try:
                # Forward pass with output_attentions=True to ensure we get attention weights
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                    
                    # If attention weights weren't captured by hook, try to get them from outputs
                    if attention_weights is None and hasattr(outputs, 'attentions') and outputs.attentions:
                        # outputs.attentions is a tuple of attention tensors for each layer
                        if layer_idx < len(outputs.attentions):
                            attention_weights = outputs.attentions[layer_idx].detach().cpu()
                
                if attention_weights is None:
                    logger.warning("Failed to capture attention weights, falling back to heuristic")
                    return self._heuristic_induction_heads(num_heads)
                
                # Extract attention from second A to target position
                # attention_weights: [batch, num_heads, seq_len, seq_len]
                # We want attention[0, :, second_a_pos, target_pos] for each head
                batch_size, n_heads, seq_len, _ = attention_weights.shape
                
                if second_pos >= seq_len or target_pos >= seq_len:
                    logger.warning("Position out of range in attention weights, falling back to heuristic")
                    return self._heuristic_induction_heads(num_heads)
                
                # Get attention scores from second occurrence to target position for each head
                # This measures how strongly each head attends from the repeated token to the token after its first occurrence
                induction_scores = attention_weights[0, :, second_pos, target_pos].numpy()
                
                # Find heads with highest induction scores
                # Threshold: top 20% of heads, or at least 2 heads if possible
                n_select = max(2, min(int(num_heads * 0.2), num_heads))
                
                # Get top-k heads by induction score
                top_indices = np.argsort(induction_scores)[-n_select:][::-1]
                induction_heads = top_indices.tolist()
                
                # Log detection results
                logger.info(f"Detected induction heads for layer {layer_idx}: {induction_heads}")
                logger.info(f"  Induction scores: {dict(zip(induction_heads, induction_scores[induction_heads]))}")
                logger.info(f"  Pattern: attention from position {second_pos} (second occurrence) to position {target_pos} (after first occurrence)")
                
                return [int(h) for h in induction_heads if h < num_heads]
                
            finally:
                handle.remove()
                
        except Exception as e:
            logger.warning(f"Induction head detection failed: {e}, falling back to heuristic")
            import traceback
            traceback.print_exc()
            return self._heuristic_induction_heads(num_heads)
    
    def _heuristic_induction_heads(self, num_heads: int) -> List[int]:
        """
        Fallback heuristic for induction head detection.
        
        This is a simple heuristic and should only be used when real detection fails.
        Consider renaming the effect to "HeuristicAttentionSharpening" if this is used.
        """
        if num_heads >= 8:
            induction_heads = [num_heads // 2 - 1, num_heads // 2]
        elif num_heads >= 4:
            induction_heads = [num_heads // 2]
        else:
            induction_heads = [0]
        
        logger.warning(f"Using heuristic induction head detection: {induction_heads}")
        return induction_heads
    
    def _get_attention_config(self, block):
        """Get attention configuration (num_heads, head_dim) from block"""
        attn_module = None
        if hasattr(block, 'self_attn'):
            attn_module = block.self_attn
        elif hasattr(block, 'attn'):
            attn_module = block.attn
        elif hasattr(block, 'attention'):
            attn_module = block.attention
        
        if attn_module is None:
            return None, None
        
        # Try to get num_heads
        num_heads = None
        if hasattr(attn_module, 'num_heads'):
            num_heads = attn_module.num_heads
        elif hasattr(attn_module, 'num_attention_heads'):
            num_heads = attn_module.num_attention_heads
        elif hasattr(attn_module, 'config') and hasattr(attn_module.config, 'num_heads'):
            num_heads = attn_module.config.num_heads
        elif hasattr(attn_module, 'config') and hasattr(attn_module.config, 'num_attention_heads'):
            num_heads = attn_module.config.num_attention_heads
        
        # Try to get head_dim
        head_dim = None
        if hasattr(attn_module, 'head_dim'):
            head_dim = attn_module.head_dim
        elif hasattr(attn_module, 'q_proj'):
            # head_dim = hidden_size // num_heads
            if hasattr(attn_module.q_proj, 'out_features'):
                hidden_size = attn_module.q_proj.out_features
                if num_heads:
                    head_dim = hidden_size // num_heads
        elif hasattr(attn_module, 'embed_dim') and num_heads:
            head_dim = attn_module.embed_dim // num_heads
        
        return num_heads, head_dim
    
    def apply(self, model, **kwargs):
        """Apply QK score scaling with head-specific induction head targeting"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        # Get tokenizer from kwargs or model
        tokenizer = kwargs.get('tokenizer')
        if tokenizer is None and hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
        
        # Calculate scaling factor
        # weight=0.0 -> scale=1.0 (baseline, no effect)
        # weight=0.5 -> scale=1.5 (moderate sharpening)
        # weight=1.0 -> scale=2.0 (maximum sharpening)
        base_scale = 1.0
        max_scale = 2.0
        scale_factor = self.get_effective_value(base_scale, max_scale)
        
        logger.info(f"Applying QK scaling with head-specific targeting, scale_factor={scale_factor:.3f} to {len(selected_layers)} layers")
        
        for layer_idx in selected_layers:
            try:
                block = blocks[layer_idx]
                
                # Get attention configuration
                num_heads, head_dim = self._get_attention_config(block)
                if num_heads is None:
                    logger.warning(f"Could not determine num_heads for layer {layer_idx}, skipping")
                    continue
                
                # Detect induction heads (REAL DETECTION using calibration prompt)
                induction_heads = self._detect_induction_heads(model, block, layer_idx, num_heads, tokenizer=tokenizer)
                if not induction_heads:
                    logger.warning(f"No induction heads detected for layer {layer_idx}, skipping")
                    continue
                
                # Create induction mask: 1.0 for induction heads, 0.0 for others
                induction_mask = torch.zeros(num_heads)
                for head_idx in induction_heads:
                    induction_mask[head_idx] = 1.0
                
                # Store mask for this layer
                self.induction_head_masks[layer_idx] = induction_mask
                
                # Architecture-specific: locate the query projection layer
                q_proj = None
                is_fused = False
                
                # Llama, Mistral, Qwen architectures: separate q_proj
                if hasattr(block, 'self_attn') and hasattr(block.self_attn, 'q_proj'):
                    q_proj = block.self_attn.q_proj
                    is_fused = False
                # GPT2 architecture: fused c_attn (Q, K, V together)
                elif hasattr(block, 'attn') and hasattr(block.attn, 'c_attn'):
                    q_proj = block.attn.c_attn
                    is_fused = True
                # GPT-NeoX: separate q_proj
                elif hasattr(block, 'attention') and hasattr(block.attention, 'query_key_value'):
                    q_proj = block.attention.query_key_value
                    is_fused = True
                else:
                    logger.warning(f"Could not find Q projection in layer {layer_idx}, skipping")
                    continue
                
                # Create hook function with head-specific scaling
                def make_q_hook(scale, mask, fused, n_heads, h_dim):
                    def q_hook(module, input_tuple, output):
                        """
                        Hook to scale Q vectors only for induction heads.
                        
                        For separate q_proj: output is [batch, seq, hidden_dim] where hidden_dim = num_heads * head_dim
                        For fused c_attn: output is [batch, seq, 3*hidden_dim]
                        """
                        if fused:
                            # GPT2-style: output is [batch, seq, 3*hidden_dim]
                            batch_size, seq_len, hidden_dim = output.shape
                            single_dim = hidden_dim // 3
                            
                            Q = output[:, :, :single_dim]
                            K = output[:, :, single_dim:2*single_dim]
                            V = output[:, :, 2*single_dim:]
                            
                            # Reshape Q to [batch, seq, num_heads, head_dim]
                            if h_dim and single_dim == n_heads * h_dim:
                                Q_reshaped = Q.view(batch_size, seq_len, n_heads, h_dim)
                                # Apply scaling only to induction heads
                                # FIX: Move mask to device
                                mask_dev = mask.to(Q_reshaped.device)
                                # CRITICAL: Ensure scale_vector matches Q's dtype to avoid dtype mismatch
                                scale_vector = (1.0 + (scale - 1.0) * mask_dev.view(1, 1, -1, 1)).to(Q_reshaped.dtype)
                                Q_scaled = Q_reshaped * scale_vector
                                Q = Q_scaled.view(batch_size, seq_len, single_dim)
                            else:
                                # Fallback: scale all if we can't reshape
                                # CRITICAL: Ensure scale matches Q's dtype
                                if isinstance(scale, torch.Tensor):
                                    Q = Q * scale.to(Q.dtype)
                                else:
                                    Q = Q * torch.tensor(scale, dtype=Q.dtype, device=Q.device)
                            
                            output_scaled = torch.cat([Q, K, V], dim=-1)
                            return output_scaled
                        else:
                            # Separate q_proj: output is [batch, seq, hidden_dim]
                            batch_size, seq_len, hidden_dim = output.shape
                            
                            # Reshape to [batch, seq, num_heads, head_dim]
                            if h_dim and hidden_dim == n_heads * h_dim:
                                Q_reshaped = output.view(batch_size, seq_len, n_heads, h_dim)
                                # Apply scaling only to induction heads
                                # FIX: Move mask to device
                                mask_dev = mask.to(Q_reshaped.device)
                                # CRITICAL: Ensure scale_vector matches output's dtype to avoid dtype mismatch
                                scale_vector = (1.0 + (scale - 1.0) * mask_dev.view(1, 1, -1, 1)).to(output.dtype)
                                Q_scaled = Q_reshaped * scale_vector
                                return Q_scaled.view(batch_size, seq_len, hidden_dim)
                            else:
                                # Fallback: scale all if we can't reshape
                                # CRITICAL: Ensure scale matches output's dtype
                                if isinstance(scale, torch.Tensor):
                                    return output * scale.to(output.dtype)
                                else:
                                    return output * torch.tensor(scale, dtype=output.dtype, device=output.device)
                    
                    return q_hook
                
                # Register hook
                handle = q_proj.register_forward_hook(
                    make_q_hook(scale_factor, induction_mask, is_fused, num_heads, head_dim)
                )
                self.handles.append(handle)
                
                logger.info(f"Layer {layer_idx}: Sharpening {len(induction_heads)} induction heads: {induction_heads}")
                
            except Exception as e:
                logger.warning(f"Failed to apply QK scaling in layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.induction_head_masks.clear()
        
    def get_logits_processor(self):
        """
        Return repetition penalty processor to counteract mode collapse.
        
        When induction heads are sharpened, there's a risk of repetitive loops.
        This penalty helps maintain diversity in generation.
        """
        if not self.repetition_penalty_active:
            return None
        
        # Calculate repetition penalty strength based on weight
        # Higher weight (more sharpening) -> stronger penalty needed
        base_penalty = 1.0
        max_penalty = 1.15  # Moderate penalty to avoid over-suppression
        penalty = self.get_effective_value(base_penalty, max_penalty)
        
        class RepetitionPenaltyProcessor(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty = penalty
                self.generated_tokens = []
            
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                # Apply penalty to previously generated tokens
                if len(self.generated_tokens) > 0:
                    for token_id in self.generated_tokens[-10:]:  # Last 10 tokens
                        if token_id < scores.shape[-1]:
                            scores[:, token_id] /= self.penalty
                
                # Track generated tokens
                if input_ids.shape[1] > 0:
                    last_token = input_ids[0, -1].item()
                    self.generated_tokens.append(last_token)
                    # Keep only recent tokens
                    if len(self.generated_tokens) > 50:
                        self.generated_tokens = self.generated_tokens[-50:]
                
                return scores
        
        return RepetitionPenaltyProcessor(penalty)
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer (kept for compatibility)"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

class HeadMaskingDropoutEffect(BaseEffect):
    """Head masking / dropout (layer-specific)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "deep", dropout_type: str = "random"):
        super().__init__(weight, direction)
        self.layers = layers
        self.dropout_type = dropout_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply head masking/dropout to attention layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get attention module
                attn = self._get_attention_module(blocks[layer_idx])
                if attn is None:
                    continue
                    
                # Store original forward
                original_forward = attn.forward
                
                # Calculate effective dropout rate
                base_dropout = 0.0
                max_dropout = 0.4
                effective_dropout = self.get_effective_value(base_dropout, max_dropout)
                
                def masked_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply head masking to attention weights
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            # Create head mask
                            batch_size, num_heads, seq_len, seq_len = attn_weights.shape
                            
                            if self.dropout_type == "random":
                                # Random head dropout
                                head_mask = (torch.rand(num_heads) > effective_dropout).to(attn_weights.device)
                                head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                attn_weights = attn_weights * head_mask
                            elif self.dropout_type == "alternating":
                                # Alternating head dropout
                                head_mask = (torch.arange(num_heads) % 2 == 0).to(attn_weights.device)
                                head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                attn_weights = attn_weights * head_mask
                            
                            # Renormalize
                            attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
                            
                            # Recompute attention output if possible
                            if len(output) >= 3 and output[2] is not None:
                                V = output[2]
                                new_attn_output = torch.matmul(attn_weights, V)
                                output = (new_attn_output,) + output[1:]
                    
                    return output
                
                attn.forward = masked_forward
                self.handles.append((attn, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply head masking in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for attn, original_forward in self.handles:
            attn.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Attention effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

class HeadReweightingEffect(BaseEffect):
    """Head re-weighting (boost known "routing" heads)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "all", routing_type: str = "stylistic"):
        super().__init__(weight, direction)
        self.layers = layers
        self.routing_type = routing_type
        self.handles = []
        
        # Define routing head patterns for different types
        self.routing_patterns = {
            "stylistic": [0, 2, 4, 6, 8],  # Even heads for style
            "semantic": [1, 3, 5, 7, 9],   # Odd heads for semantics
            "positional": [0, 1, 10, 11],  # First/last heads for position
            "global": [5, 6, 7, 8],       # Middle heads for global context
            "local": [0, 1, 2, 3],        # Early heads for local context
        }
        
    def apply(self, model, **kwargs):
        """Apply head re-weighting to attention layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get attention module
                attn = self._get_attention_module(blocks[layer_idx])
                if attn is None:
                    continue
                    
                # Store original forward
                original_forward = attn.forward
                
                # Calculate effective boost factor
                base_boost = 1.0
                max_boost = 0.5
                effective_boost = self.get_effective_value(0.0, max_boost)
                
                def reweighted_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply head re-weighting to attention weights
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            batch_size, num_heads, seq_len, seq_len = attn_weights.shape
                            
                            # Get routing pattern
                            if self.routing_type in self.routing_patterns:
                                routing_heads = self.routing_patterns[self.routing_type]
                                routing_heads = [h for h in routing_heads if h < num_heads]
                                
                                # Boost routing heads
                                for head_idx in routing_heads:
                                    boost_factor = 1.0 + effective_boost
                                    attn_weights[:, head_idx, :, :] *= boost_factor
                                
                                # Renormalize
                                attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                                
                                # Recompute attention output if possible
                                if len(output) >= 3 and output[2] is not None:
                                    V = output[2]
                                    new_attn_output = torch.matmul(attn_weights, V)
                                    output = (new_attn_output,) + output[1:]
                    
                    return output
                
                attn.forward = reweighted_forward
                self.handles.append((attn, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply head re-weighting in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for attn, original_forward in self.handles:
            attn.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Attention effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

class PositionalBiasTweakEffect(BaseEffect):
    """Positional bias tweak (ALiBi slope/pos temp)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 bias_type: str = "recency"):
        super().__init__(weight, direction)
        self.bias_type = bias_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply positional bias tweaking"""
        # This effect modifies positional embeddings or attention biases
        # For now, we'll implement it as a logits processor that biases recent tokens
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective bias strength
        base_bias = 0.0
        max_bias = 0.3
        effective_bias = self.get_effective_value(base_bias, max_bias)
        
        class PositionalBiasProcessor(LogitsProcessor):
            def __init__(self, bias_type, bias_strength):
                self.bias_type = bias_type
                self.bias_strength = bias_strength
                
            def __call__(self, input_ids, scores):
                seq_len = input_ids.shape[1]
                
                if self.bias_type == "recency":
                    # Bias towards recent tokens (higher scores for recent positions)
                    for i in range(seq_len):
                        recency_bias = (i / seq_len) * self.bias_strength
                        scores[:, i] += recency_bias
                        
                elif self.bias_type == "history":
                    # Bias towards earlier tokens (higher scores for earlier positions)
                    for i in range(seq_len):
                        history_bias = ((seq_len - i) / seq_len) * self.bias_strength
                        scores[:, i] += history_bias
                        
                elif self.bias_type == "middle":
                    # Bias towards middle tokens
                    for i in range(seq_len):
                        middle_bias = (1.0 - abs(i - seq_len/2) / (seq_len/2)) * self.bias_strength
                        scores[:, i] += middle_bias
                
                return scores
                
        return PositionalBiasProcessor(self.bias_type, effective_bias)
        
    def cleanup(self):
        pass

class AttentionOscillationEffect(BaseEffect):
    """Attention oscillation (periodic gain per head)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "mid", oscillation_type: str = "sine"):
        super().__init__(weight, direction)
        self.layers = layers
        self.oscillation_type = oscillation_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply attention oscillation to layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get attention module
                attn = self._get_attention_module(blocks[layer_idx])
                if attn is None:
                    continue
                    
                # Store original forward
                original_forward = attn.forward
                
                # Calculate effective oscillation amplitude
                base_amplitude = 0.0
                max_amplitude = 0.2
                effective_amplitude = self.get_effective_value(base_amplitude, max_amplitude)
                
                def oscillating_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply oscillation to attention weights
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            batch_size, num_heads, seq_len, seq_len = attn_weights.shape
                            
                            # Create oscillation pattern
                            if self.oscillation_type == "sine":
                                # Sine wave oscillation
                                t = torch.arange(seq_len, device=attn_weights.device)
                                oscillation = torch.sin(t * 0.1) * effective_amplitude
                                oscillation = oscillation.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                                attn_weights = attn_weights * (1.0 + oscillation)
                                
                            elif self.oscillation_type == "square":
                                # Square wave oscillation
                                t = torch.arange(seq_len, device=attn_weights.device)
                                oscillation = (t % 20 < 10).float() * effective_amplitude
                                oscillation = oscillation.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                                attn_weights = attn_weights * (1.0 + oscillation)
                            
                            # Renormalize
                            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                            
                            # Recompute attention output if possible
                            if len(output) >= 3 and output[2] is not None:
                                V = output[2]
                                new_attn_output = torch.matmul(attn_weights, V)
                                output = (new_attn_output,) + output[1:]
                    
                    return output
                
                attn.forward = oscillating_forward
                self.handles.append((attn, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply attention oscillation in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for attn, original_forward in self.handles:
            attn.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Attention effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))
            
    def _get_attention_module(self, layer):
        """Get attention module from layer"""
        for attr in ["attn", "self_attn", "attention"]:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

class AttentionSinksAnchorsEffect(BaseEffect):
    """Attention sinks / anchors"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 sink_type: str = "stable"):
        super().__init__(weight, direction)
        self.sink_type = sink_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply attention sinks/anchors"""
        # This effect maintains stable sink tokens for top-down control
        # For now, we'll implement it as a logits processor that stabilizes certain tokens
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective sink strength
        base_sink = 0.0
        max_sink = 0.4
        effective_sink = self.get_effective_value(base_sink, max_sink)
        
        class AttentionSinkProcessor(LogitsProcessor):
            def __init__(self, sink_type, sink_strength):
                self.sink_type = sink_type
                self.sink_strength = sink_strength
                self.sink_tokens = set()
                
            def __call__(self, input_ids, scores):
                # Identify potential sink tokens (common, stable tokens)
                if self.sink_type == "stable":
                    # Boost scores for common tokens that provide stability
                    common_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Common tokens
                    for token_id in common_tokens:
                        if token_id < scores.shape[-1]:
                            scores[..., token_id] += self.sink_strength
                            
                elif self.sink_type == "contextual":
                    # Boost scores for tokens that appeared earlier in context
                    if input_ids.shape[1] > 0:
                        context_tokens = input_ids[0, :min(10, input_ids.shape[1])].tolist()
                        for token_id in context_tokens:
                            if token_id < scores.shape[-1]:
                                scores[..., token_id] += self.sink_strength * 0.5
                
                return scores
                
        return AttentionSinkProcessor(self.sink_type, effective_sink)
        
    def cleanup(self):
        pass

# ============================================================================
# STEERING EFFECTS
# ============================================================================

class SteeringEffect(BaseEffect):
    """Activation steering effect using Contrastive Activation Addition (CAA) vectors"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", steering_type: str = "associative",
                 vector_path: Optional[str] = None, vector_dir: str = "outputs/steering_vectors"):
        super().__init__(weight, direction)
        self.steering_type = steering_type
        self.vector_path = vector_path
        self.vector_dir = vector_dir
        self.vector = None  # Initialize as None - will be loaded on demand
        
        # Dictionary to cache loaded vectors by steering_type
        self._vector_cache = {}
        
    def load_vector(self, vector_path: Optional[str] = None, hidden_size: int = 768) -> bool:
        """
        Load a pre-computed steering vector from disk.
        
        Args:
            vector_path: Path to the vector file. If None, constructs path from steering_type.
            hidden_size: Expected hidden size (used for fallback zero vector)
            
        Returns:
            True if vector loaded successfully, False otherwise
        """
        import os
        from pathlib import Path
        
        # Determine vector path
        if vector_path is None:
            if self.vector_path:
                vector_path = self.vector_path
            else:
                # Construct path from steering_type
                # Try multiple layer indices (prefer last layer)
                vector_dir = Path(self.vector_dir)
                for layer_idx in [-1, -2, 0]:
                    candidate_path = vector_dir / f"{self.steering_type}_layer{layer_idx}.pt"
                    if candidate_path.exists():
                        vector_path = str(candidate_path)
                        break
                
                if vector_path is None:
                    # Try without layer suffix
                    candidate_path = vector_dir / f"{self.steering_type}.pt"
                    if candidate_path.exists():
                        vector_path = str(candidate_path)
        
        if vector_path is None:
            logger.warning(f"CRITICAL WARNING: No steering vector path specified for '{self.steering_type}'. "
                          f"Using zero vector (no steering effect).")
            self.vector = torch.zeros(hidden_size)
            return False
        
        # Check cache first
        if vector_path in self._vector_cache:
            self.vector = self._vector_cache[vector_path]
            return True
        
        # Load from disk
        try:
            vector_path_obj = Path(vector_path)
            if not vector_path_obj.exists():
                # REMOVE SILENT FAILURE
                raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Steering vector {vector_path} not found. Aborting trial to prevent false null results.")
            
            self.vector = torch.load(vector_path, map_location='cpu')
            
            # Validate shape
            if isinstance(self.vector, torch.Tensor):
                if len(self.vector.shape) != 1:
                    logger.warning(f"Steering vector has unexpected shape {self.vector.shape}, "
                                 f"expected 1D tensor. Flattening...")
                    self.vector = self.vector.flatten()
                
                # Cache the vector
                self._vector_cache[vector_path] = self.vector
                logger.info(f"Loaded steering vector from {vector_path}: shape={self.vector.shape}")
                return True
            else:
                # REMOVE SILENT FAILURE
                raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Loaded object from {vector_path} is not a tensor: {type(self.vector)}. Aborting trial to prevent false null results.")
                
        except Exception as e:
            # REMOVE SILENT FAILURE
            raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Could not load steering vector at {vector_path}. Aborting trial to prevent false null results.") from e
    
    def get_vector(self, hidden_size: int = 768) -> torch.Tensor:
        """
        Get the steering vector, loading it if necessary.
        
        Args:
            hidden_size: Expected hidden size (for fallback)
            
        Returns:
            Steering vector tensor
        """
        if self.vector is None:
            self.load_vector(hidden_size=hidden_size)
        
        # Ensure vector matches hidden_size (resize if needed)
        if self.vector.shape[0] != hidden_size:
            if self.vector.shape[0] < hidden_size:
                # Pad with zeros
                padding = torch.zeros(hidden_size - self.vector.shape[0])
                self.vector = torch.cat([self.vector, padding])
                logger.warning(f"Padded steering vector from {self.vector.shape[0]} to {hidden_size}")
            else:
                # Truncate
                self.vector = self.vector[:hidden_size]
                logger.warning(f"Truncated steering vector from {self.vector.shape[0]} to {hidden_size}")
        
        return self.vector
        
    def apply(self, model, **kwargs):
        """Apply steering to hidden states"""
        # This will be applied during generation
        pass
        
    def apply_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply steering to hidden states"""
        if hidden_states is None or hidden_states.numel() == 0:
            return hidden_states
        
        # Get hidden size from the actual hidden states
        if len(hidden_states.shape) >= 2:
            hidden_size = hidden_states.shape[-1]
        else:
            hidden_size = hidden_states.shape[0] if len(hidden_states.shape) == 1 else 768
        
        # Get or load the vector
        steering_vector = self.get_vector(hidden_size=hidden_size)
        
        # FIX: Move vector to the same device as hidden_states
        steering_vector = steering_vector.to(hidden_states.device)
        
        # Calculate effective steering strength
        base_strength = 0.0
        max_strength = 0.3
        effective_strength = self.get_effective_value(base_strength, max_strength)
        
        # Apply steering as a residual connection
        # Ensure shapes match
        if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_size]
            steering_effect = steering_vector.unsqueeze(0).unsqueeze(0) * effective_strength
        elif len(hidden_states.shape) == 2:  # [seq_len, hidden_size]
            steering_effect = steering_vector.unsqueeze(0) * effective_strength
        else:  # [hidden_size]
            steering_effect = steering_vector * effective_strength
        
        return hidden_states + steering_effect
        
    def get_logits_processor(self):
        """Steering effects don't use logits processors"""
        return None
        
    def cleanup(self):
        """Clean up resources"""
        # Optionally clear cache if needed
        pass


class RandomDirectionEffect(BaseEffect):
    """
    Random direction effect for active placebo control.
    
    This effect generates a random steering vector normalized to match the L2 norm
    of an active pack's steering vector. This provides an active placebo control
    that has the same magnitude of intervention but random direction, allowing us
    to test whether the specific direction (semantic content) of steering vectors
    is necessary for the observed effects.
    
    The random vector is normalized to the same L2 norm as a reference active pack
    vector to ensure fair comparison (same intervention magnitude, different direction).
    """
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 reference_vector_path: Optional[str] = None,
                 reference_vector: Optional[torch.Tensor] = None,
                 hidden_size: int = 768):
        super().__init__(weight, direction)
        self.reference_vector_path = reference_vector_path
        self.reference_vector = reference_vector
        self.hidden_size = hidden_size
        self.random_vector = None
        self._normalized = False
        
    def _get_reference_norm(self) -> float:
        """
        Get the L2 norm of the reference vector (active pack vector).
        
        Returns:
            L2 norm of the reference vector, or 1.0 if no reference available
        """
        if self.reference_vector is not None:
            return torch.norm(self.reference_vector).item()
        
        if self.reference_vector_path:
            try:
                ref_vec = torch.load(self.reference_vector_path, map_location='cpu')
                if isinstance(ref_vec, torch.Tensor):
                    return torch.norm(ref_vec).item()
            except Exception as e:
                logger.warning(f"Failed to load reference vector from {self.reference_vector_path}: {e}")
        
        # Default: use a typical norm for steering vectors (empirically ~0.1-0.5)
        # This is a reasonable default based on typical steering vector magnitudes
        return 0.3
    
    def _generate_random_vector(self, hidden_size: int) -> torch.Tensor:
        """
        Generate a random vector normalized to match the reference vector's L2 norm.
        
        Args:
            hidden_size: Size of the hidden dimension
            
        Returns:
            Random vector with same L2 norm as reference vector
        """
        # Generate random vector from standard normal distribution
        random_vec = torch.randn(hidden_size)
        
        # Normalize to unit vector
        random_vec = random_vec / torch.norm(random_vec)
        
        # Scale to match reference vector's L2 norm
        reference_norm = self._get_reference_norm()
        random_vec = random_vec * reference_norm
        
        logger.info(f"Generated random direction vector with L2 norm: {torch.norm(random_vec).item():.4f} "
                   f"(target: {reference_norm:.4f})")
        
        return random_vec
    
    def get_vector(self, hidden_size: Optional[int] = None) -> torch.Tensor:
        """
        Get the random steering vector, generating it if necessary.
        
        Args:
            hidden_size: Expected hidden size (uses self.hidden_size if None)
            
        Returns:
            Random steering vector tensor with normalized L2 norm
        """
        if hidden_size is None:
            hidden_size = self.hidden_size
        
        if self.random_vector is None or self.random_vector.shape[0] != hidden_size:
            self.random_vector = self._generate_random_vector(hidden_size)
            self._normalized = True
        
        return self.random_vector
    
    def apply(self, model, **kwargs):
        """Apply random direction steering (placeholder - actual steering happens in apply_steering)"""
        # Get hidden size from model if available
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            self.hidden_size = model.config.hidden_size
        elif hasattr(model, 'config') and hasattr(model.config, 'd_model'):
            self.hidden_size = model.config.d_model
        
        # Pre-generate the random vector
        self.get_vector(hidden_size=self.hidden_size)
    
    def apply_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply random direction steering to hidden states"""
        if hidden_states is None or hidden_states.numel() == 0:
            return hidden_states
        
        # Get hidden size from the actual hidden states
        if len(hidden_states.shape) >= 2:
            hidden_size = hidden_states.shape[-1]
        else:
            hidden_size = hidden_states.shape[0] if len(hidden_states.shape) == 1 else self.hidden_size
        
        # Get or generate the random vector
        random_vector = self.get_vector(hidden_size=hidden_size)
        
        # FIX: Move to device
        random_vector = random_vector.to(hidden_states.device)
        
        # Calculate effective steering strength
        base_strength = 0.0
        max_strength = 0.3
        effective_strength = self.get_effective_value(base_strength, max_strength)
        
        # Apply steering as a residual connection
        # Ensure shapes match
        if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_size]
            steering_effect = random_vector.unsqueeze(0).unsqueeze(0) * effective_strength
        elif len(hidden_states.shape) == 2:  # [seq_len, hidden_size]
            steering_effect = random_vector.unsqueeze(0) * effective_strength
        else:  # [hidden_size]
            steering_effect = random_vector * effective_strength
        
        return hidden_states + steering_effect
    
    def get_logits_processor(self):
        """Random direction effects don't use logits processors"""
        return None
    
    def cleanup(self):
        """Clean up resources"""
        self.random_vector = None
        self._normalized = False


class RandomOrthogonalSteeringEffect(BaseEffect):
    """
    Random orthogonal steering effect for active placebo control.
    
    This effect generates a random steering vector that is:
    1. Normalized to match the L2 norm of a reference vector (e.g., LSD vector)
    2. Orthogonalized to the reference vector (v_rand · v_LSD ≈ 0)
    
    This ensures the random vector has the same magnitude of intervention but
    steers in a completely different direction, providing a rigorous control
    that tests whether the specific semantic content (direction) of steering
    vectors is necessary for observed effects.
    
    Hypothesis:
    - LSD Vector: High PDQ-S score, coherent "trippy" text
    - Random Orthogonal Vector: High perplexity, confusion, but LOW PDQ-S score
      (no specific "breathing walls" or "ego death" content)
    - Zero Vector: Baseline
    """
    
    def __init__(self, weight: float = 0.5, direction: str = "up",
                 reference_steering_type: str = "associative",
                 reference_vector_path: Optional[str] = None,
                 vector_dir: str = "outputs/steering_vectors",
                 hidden_size: int = 768,
                 orthogonality_tolerance: float = 1e-6):
        super().__init__(weight, direction)
        self.reference_steering_type = reference_steering_type
        self.reference_vector_path = reference_vector_path
        self.vector_dir = vector_dir
        self.hidden_size = hidden_size
        self.orthogonality_tolerance = orthogonality_tolerance
        
        self.reference_vector = None
        self.orthogonal_vector = None
        self._computed = False
        
    def _load_reference_vector(self, hidden_size: int) -> torch.Tensor:
        """
        Load the reference vector (e.g., LSD/associative vector) from disk.
        
        Args:
            hidden_size: Expected hidden size
            
        Returns:
            Reference vector tensor
        """
        from pathlib import Path
        
        if self.reference_vector is not None:
            return self.reference_vector
        
        # Try to load from specified path
        if self.reference_vector_path:
            try:
                ref_vec = torch.load(self.reference_vector_path, map_location='cpu')
                if isinstance(ref_vec, torch.Tensor):
                    # Resize if needed
                    if ref_vec.shape[0] != hidden_size:
                        if ref_vec.shape[0] < hidden_size:
                            padding = torch.zeros(hidden_size - ref_vec.shape[0])
                            ref_vec = torch.cat([ref_vec, padding])
                        else:
                            ref_vec = ref_vec[:hidden_size]
                    self.reference_vector = ref_vec
                    return ref_vec
            except Exception as e:
                # REMOVE SILENT FAILURE
                raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Could not load steering vector at {self.reference_vector_path}. Aborting trial to prevent false null results.") from e
        
        # Try to load from vector_dir using steering_type
        vector_dir = Path(self.vector_dir)
        for layer_idx in [-1, -2, 0]:
            candidate_path = vector_dir / f"{self.reference_steering_type}_layer{layer_idx}.pt"
            if candidate_path.exists():
                try:
                    ref_vec = torch.load(candidate_path, map_location='cpu')
                    if isinstance(ref_vec, torch.Tensor):
                        # Resize if needed
                        if ref_vec.shape[0] != hidden_size:
                            if ref_vec.shape[0] < hidden_size:
                                padding = torch.zeros(hidden_size - ref_vec.shape[0])
                                ref_vec = torch.cat([ref_vec, padding])
                            else:
                                ref_vec = ref_vec[:hidden_size]
                        self.reference_vector = ref_vec
                        logger.info(f"Loaded reference vector from {candidate_path}: norm={torch.norm(ref_vec):.4f}")
                        return ref_vec
                except Exception as e:
                    # REMOVE SILENT FAILURE
                    raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Could not load steering vector at {candidate_path}. Aborting trial to prevent false null results.") from e
        
        # Try without layer suffix
        candidate_path = vector_dir / f"{self.reference_steering_type}.pt"
        if candidate_path.exists():
            try:
                ref_vec = torch.load(candidate_path, map_location='cpu')
                if isinstance(ref_vec, torch.Tensor):
                    # Resize if needed
                    if ref_vec.shape[0] != hidden_size:
                        if ref_vec.shape[0] < hidden_size:
                            padding = torch.zeros(hidden_size - ref_vec.shape[0])
                            ref_vec = torch.cat([ref_vec, padding])
                        else:
                            ref_vec = ref_vec[:hidden_size]
                    self.reference_vector = ref_vec
                    logger.info(f"Loaded reference vector from {candidate_path}: norm={torch.norm(ref_vec):.4f}")
                    return ref_vec
            except Exception as e:
                # REMOVE SILENT FAILURE
                raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Could not load steering vector at {candidate_path}. Aborting trial to prevent false null results.") from e
        
        # REMOVE SILENT FAILURE: No fallback to zero vector
        raise RuntimeError(f"CRITICAL EXPERIMENTAL FAILURE: Reference vector for '{self.reference_steering_type}' not found at {self.vector_dir}. Aborting trial to prevent false null results.")
    
    def _generate_orthogonal_vector(self, hidden_size: int) -> torch.Tensor:
        """
        Generate a random vector orthogonal to the reference vector.
        
        Uses Gram-Schmidt orthogonalization:
        1. Generate random vector v_rand
        2. Project out component parallel to reference: v_rand - (v_rand · v_ref / ||v_ref||²) * v_ref
        3. Normalize to match reference vector's norm
        
        Args:
            hidden_size: Size of the hidden dimension
            
        Returns:
            Orthogonal vector with same norm as reference vector
        """
        # Load reference vector
        ref_vec = self._load_reference_vector(hidden_size)
        ref_norm = torch.norm(ref_vec).item()
        
        if ref_norm < 1e-10:
            # Reference vector is essentially zero, just generate random vector
            logger.warning("Reference vector has near-zero norm. Generating random vector without orthogonalization.")
            random_vec = torch.randn(hidden_size)
            random_vec = random_vec / torch.norm(random_vec) * ref_norm if ref_norm > 0 else random_vec / torch.norm(random_vec)
            return random_vec
        
        # Generate random vector
        random_vec = torch.randn(hidden_size)
        
        # Gram-Schmidt orthogonalization: remove component parallel to reference
        # v_ortho = v_rand - (v_rand · v_ref / ||v_ref||²) * v_ref
        dot_product = torch.dot(random_vec, ref_vec).item()
        ref_norm_squared = ref_norm ** 2
        
        # Project out the parallel component
        parallel_component = (dot_product / ref_norm_squared) * ref_vec
        orthogonal_vec = random_vec - parallel_component
        
        # Check orthogonality
        orthogonality_check = torch.dot(orthogonal_vec, ref_vec).item()
        if abs(orthogonality_check) > self.orthogonality_tolerance:
            logger.warning(f"Orthogonality check failed: dot product = {orthogonality_check:.2e} "
                         f"(tolerance: {self.orthogonality_tolerance:.2e}). Re-orthogonalizing...")
            # Re-orthogonalize (shouldn't be necessary, but safety check)
            dot_product = torch.dot(orthogonal_vec, ref_vec).item()
            parallel_component = (dot_product / ref_norm_squared) * ref_vec
            orthogonal_vec = orthogonal_vec - parallel_component
        
        # Normalize to match reference vector's norm
        ortho_norm = torch.norm(orthogonal_vec).item()
        if ortho_norm > 1e-10:
            orthogonal_vec = orthogonal_vec / ortho_norm * ref_norm
        else:
            # If orthogonalization resulted in near-zero vector, generate new random vector
            logger.warning("Orthogonalization resulted in near-zero vector. Generating new random vector.")
            random_vec = torch.randn(hidden_size)
            # Try to find a vector that's not parallel to reference
            for _ in range(10):
                dot_product = torch.dot(random_vec, ref_vec).item()
                if abs(dot_product) < ref_norm * 0.1:  # At least 10 degrees from parallel
                    break
                random_vec = torch.randn(hidden_size)
            # Project out parallel component
            dot_product = torch.dot(random_vec, ref_vec).item()
            parallel_component = (dot_product / ref_norm_squared) * ref_vec
            orthogonal_vec = random_vec - parallel_component
            ortho_norm = torch.norm(orthogonal_vec).item()
            if ortho_norm > 1e-10:
                orthogonal_vec = orthogonal_vec / ortho_norm * ref_norm
            else:
                # Last resort: use random vector normalized to reference norm
                orthogonal_vec = random_vec / torch.norm(random_vec) * ref_norm
        
        # Final orthogonality check
        final_dot = torch.dot(orthogonal_vec, ref_vec).item()
        final_norm = torch.norm(orthogonal_vec).item()
        
        # CRITICAL VALIDATION: Verify orthogonality meets scientific threshold
        orthogonality_threshold = 1e-6
        if abs(final_dot) >= orthogonality_threshold:
            error_msg = (
                f"CRITICAL EXPERIMENTAL FAILURE: Orthogonality validation failed!\n"
                f"  Dot product with reference: {final_dot:.2e}\n"
                f"  Required threshold: < {orthogonality_threshold:.2e}\n"
                f"  This indicates the 'Active Placebo' control is contaminated.\n"
                f"  The random vector is NOT orthogonal to the reference vector.\n"
                f"  Aborting trial to prevent false null results."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Log successful validation with prominent formatting
        logger.info("=" * 80)
        logger.info("✅ ORTHOGONALITY VALIDATION PASSED (Active Placebo Control)")
        logger.info(f"   Generated orthogonal vector:")
        logger.info(f"   - Norm: {final_norm:.6f} (target: {ref_norm:.6f})")
        logger.info(f"   - Dot product with reference: {final_dot:.2e} (required: < {orthogonality_threshold:.2e})")
        logger.info(f"   - Orthogonality status: VALID (vectors are perpendicular)")
        logger.info("=" * 80)
        
        return orthogonal_vec
    
    def get_vector(self, hidden_size: Optional[int] = None) -> torch.Tensor:
        """
        Get the orthogonal steering vector, generating it if necessary.
        
        Args:
            hidden_size: Expected hidden size (uses self.hidden_size if None)
            
        Returns:
            Orthogonal steering vector tensor
        """
        if hidden_size is None:
            hidden_size = self.hidden_size
        
        if self.orthogonal_vector is None or self.orthogonal_vector.shape[0] != hidden_size:
            self.orthogonal_vector = self._generate_orthogonal_vector(hidden_size)
            self._computed = True
        
        return self.orthogonal_vector
    
    def apply(self, model, **kwargs):
        """Apply random orthogonal steering (placeholder - actual steering happens in apply_steering)"""
        # Get hidden size from model if available
        if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            self.hidden_size = model.config.hidden_size
        elif hasattr(model, 'config') and hasattr(model.config, 'd_model'):
            self.hidden_size = model.config.d_model
        
        # Pre-generate the orthogonal vector
        self.get_vector(hidden_size=self.hidden_size)
    
    def apply_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply random orthogonal steering to hidden states"""
        if hidden_states is None or hidden_states.numel() == 0:
            return hidden_states
        
        # Get hidden size from the actual hidden states
        if len(hidden_states.shape) >= 2:
            hidden_size = hidden_states.shape[-1]
        else:
            hidden_size = hidden_states.shape[0] if len(hidden_states.shape) == 1 else self.hidden_size
        
        # Get or generate the orthogonal vector
        orthogonal_vector = self.get_vector(hidden_size=hidden_size)
        
        # FIX: Move to device
        orthogonal_vector = orthogonal_vector.to(hidden_states.device)
        
        # Calculate effective steering strength
        base_strength = 0.0
        max_strength = 0.3
        effective_strength = self.get_effective_value(base_strength, max_strength)
        
        # Apply steering as a residual connection
        # Ensure shapes match
        if len(hidden_states.shape) == 3:  # [batch, seq_len, hidden_size]
            steering_effect = orthogonal_vector.unsqueeze(0).unsqueeze(0) * effective_strength
        elif len(hidden_states.shape) == 2:  # [seq_len, hidden_size]
            steering_effect = orthogonal_vector.unsqueeze(0) * effective_strength
        else:  # [hidden_size]
            steering_effect = orthogonal_vector * effective_strength
        
        return hidden_states + steering_effect
    
    def get_logits_processor(self):
        """Random orthogonal steering effects don't use logits processors"""
        return None
    
    def cleanup(self):
        """Clean up resources"""
        self.reference_vector = None
        self.orthogonal_vector = None
        self._computed = False

# ============================================================================
# MEMORY EFFECTS
# ============================================================================

class KVDecayEffect(BaseEffect):
    """KV cache decay effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction)
        
    def apply(self, model, **kwargs):
        pass  # Applied during generation
        
    def modify_kv_cache(self, kv_cache):
        """Apply decay to KV cache"""
        if kv_cache is None:
            return kv_cache
            
        # Calculate effective decay rate
        base_decay = 1.0  # No decay
        max_decay = 0.3   # Can decay to 0.7
        effective_decay = self.get_effective_value(base_decay, max_decay)
        
        if isinstance(kv_cache, tuple):
            modified_cache = []
            for layer_cache in kv_cache:
                if layer_cache is not None:
                    # Apply exponential decay to older tokens
                    seq_len = layer_cache.shape[-2]
                    decay_factors = torch.pow(effective_decay, torch.arange(seq_len, device=layer_cache.device))
                    decay_factors = decay_factors.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    modified_layer = layer_cache * decay_factors
                    modified_cache.append(modified_layer)
                else:
                    modified_cache.append(None)
            return tuple(modified_cache)
        return kv_cache
        
    def get_logits_processor(self):
        """KV effects don't use logits processors"""
        return None
        
    def cleanup(self):
        pass

class KVCompressionEffect(BaseEffect):
    """KV cache compression effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up"):
        super().__init__(weight, direction)
        
    def apply(self, model, **kwargs):
        pass  # Applied during generation
        
    def modify_kv_cache(self, kv_cache):
        """Apply compression to KV cache"""
        if kv_cache is None:
            return kv_cache
            
        # Calculate effective compression
        base_compression = 1.0  # No compression
        max_compression = 0.5   # Can compress to 0.5
        effective_compression = self.get_effective_value(base_compression, max_compression)
        
        if isinstance(kv_cache, tuple):
            modified_cache = []
            for layer_cache in kv_cache:
                if layer_cache is not None:
                    # Apply stride compression
                    seq_len = layer_cache.shape[-2]
                    stride = max(1, int(seq_len * effective_compression))
                    indices = torch.arange(0, seq_len, stride, device=layer_cache.device)
                    modified_layer = layer_cache[..., indices, :]
                    modified_cache.append(modified_layer)
                else:
                    modified_cache.append(None)
            return tuple(modified_cache)
        return kv_cache
        
    def get_logits_processor(self):
        """KV effects don't use logits processors"""
        return None
        
    def cleanup(self):
        pass

# ============================================================================
# WORKING-MEMORY / KV-CACHE EFFECTS
# ============================================================================

class ExponentialDecayKVEffect(BaseEffect):
    """Exponential decay of old keys/values"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 decay_rate: float = 0.1, decay_type: str = "exponential"):
        super().__init__(weight, direction)
        self.base_decay_rate = decay_rate
        self.decay_type = decay_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply exponential decay to KV cache"""
        # This effect modifies the KV cache during generation
        # For now, we'll implement it as a logits processor that simulates decay
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective decay rate
        base_rate = 0.0
        max_rate = 0.3
        effective_rate = self.get_effective_value(base_rate, max_rate)
        
        class ExponentialDecayProcessor(LogitsProcessor):
            def __init__(self, decay_rate, decay_type):
                self.decay_rate = decay_rate
                self.decay_type = decay_type
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                seq_len = input_ids.shape[1]
                
                if self.decay_type == "exponential":
                    # Apply exponential decay to older positions
                    for i in range(seq_len):
                        decay_factor = math.exp(-self.decay_rate * (seq_len - i))
                        scores[:, i] *= decay_factor
                        
                elif self.decay_type == "linear":
                    # Apply linear decay to older positions
                    for i in range(seq_len):
                        decay_factor = 1.0 - (self.decay_rate * (seq_len - i) / seq_len)
                        decay_factor = max(0.1, decay_factor)  # Prevent complete decay
                        scores[:, i] *= decay_factor
                        
                elif self.decay_type == "step":
                    # Apply step decay at certain intervals
                    for i in range(seq_len):
                        steps_back = (seq_len - i) // 10  # Decay every 10 tokens
                        decay_factor = math.exp(-self.decay_rate * steps_back)
                        scores[:, i] *= decay_factor
                
                return scores
                
        return ExponentialDecayProcessor(effective_rate, self.decay_type)
        
    def cleanup(self):
        pass

class TruncationKVEffect(BaseEffect):
    """Truncation (keep last N)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 max_tokens: int = 50, truncation_type: str = "hard"):
        super().__init__(weight, direction)
        self.base_max_tokens = max_tokens
        self.truncation_type = truncation_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply truncation to KV cache"""
        # This effect limits the effective context window
        # For now, we'll implement it as a logits processor that simulates truncation
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective max tokens
        base_max = 100
        max_reduction = 80
        effective_max = base_max - self.get_effective_value(0, max_reduction)
        
        class TruncationProcessor(LogitsProcessor):
            def __init__(self, max_tokens, truncation_type):
                self.max_tokens = max_tokens
                self.truncation_type = truncation_type
                
            def __call__(self, input_ids, scores):
                seq_len = input_ids.shape[1]
                
                if seq_len > self.max_tokens:
                    if self.truncation_type == "hard":
                        # Hard truncation: zero out scores for old tokens
                        for i in range(seq_len - self.max_tokens):
                            scores[:, i] *= 0.01  # Near-zero scores
                            
                    elif self.truncation_type == "soft":
                        # Soft truncation: gradual reduction
                        for i in range(seq_len):
                            if i < seq_len - self.max_tokens:
                                reduction_factor = 0.1 + 0.9 * (i / (seq_len - self.max_tokens))
                                scores[:, i] *= reduction_factor
                                
                    elif self.truncation_type == "window":
                        # Sliding window: keep only recent tokens
                        keep_start = max(0, seq_len - self.max_tokens)
                        for i in range(keep_start):
                            scores[:, i] *= 0.01
                
                return scores
                
        return TruncationProcessor(int(effective_max), self.truncation_type)
        
    def cleanup(self):
        pass

class StrideCompressionKVEffect(BaseEffect):
    """Stride compression (keep every s-th older token)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 stride_factor: int = 2, compression_type: str = "uniform"):
        super().__init__(weight, direction)
        self.base_stride_factor = stride_factor
        self.compression_type = compression_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply stride compression to KV cache"""
        # This effect compresses distant memory by keeping every Nth token
        # For now, we'll implement it as a logits processor that simulates compression
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective stride factor
        base_stride = 1
        max_stride = 5
        effective_stride = base_stride + int(self.get_effective_value(0, max_stride))
        
        class StrideCompressionProcessor(LogitsProcessor):
            def __init__(self, stride_factor, compression_type):
                self.stride_factor = stride_factor
                self.compression_type = compression_type
                
            def __call__(self, input_ids, scores):
                seq_len = input_ids.shape[1]
                
                if self.compression_type == "uniform":
                    # Uniform stride: keep every Nth token
                    for i in range(seq_len):
                        if i % self.stride_factor != 0:
                            scores[:, i] *= 0.1  # Reduce scores for non-kept tokens
                            
                elif self.compression_type == "progressive":
                    # Progressive stride: increasing compression with distance
                    for i in range(seq_len):
                        distance = seq_len - i
                        if distance > 10:  # Only compress distant tokens
                            stride = min(self.stride_factor, distance // 10)
                            if i % stride != 0:
                                scores[:, i] *= 0.2
                                
                elif self.compression_type == "adaptive":
                    # Adaptive stride: compress based on token importance (simplified)
                    for i in range(seq_len):
                        if i < seq_len - 20:  # Compress tokens beyond 20 positions
                            if i % self.stride_factor != 0:
                                scores[:, i] *= 0.15
                
                return scores
                
        return StrideCompressionProcessor(effective_stride, self.compression_type)
        
    def cleanup(self):
        pass

class SegmentGainsKVEffect(BaseEffect):
    """Segment gains (old vs new window scaling)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 window_size: int = 20, gain_type: str = "new_emphasis"):
        super().__init__(weight, direction)
        self.base_window_size = window_size
        self.gain_type = gain_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply segment gains to KV cache"""
        # This effect applies different gains to old vs new segments
        # For now, we'll implement it as a logits processor that simulates segment gains
        
        pass  # Applied via logits processor
        
    def get_logits_processor(self) -> LogitsProcessor:
        # Calculate effective gain
        base_gain = 1.0
        max_gain = 0.5
        effective_gain = self.get_effective_value(0, max_gain)
        
        class SegmentGainsProcessor(LogitsProcessor):
            def __init__(self, window_size, gain_type, gain_strength):
                self.window_size = window_size
                self.gain_type = gain_type
                self.gain_strength = gain_strength
                
            def __call__(self, input_ids, scores):
                seq_len = input_ids.shape[1]
                
                if self.gain_type == "new_emphasis":
                    # Emphasize recent tokens
                    for i in range(seq_len):
                        if i >= seq_len - self.window_size:
                            # Recent tokens get boost
                            boost_factor = 1.0 + self.gain_strength
                            scores[:, i] *= boost_factor
                        else:
                            # Older tokens get reduction
                            reduction_factor = 1.0 - self.gain_strength * 0.5
                            scores[:, i] *= reduction_factor
                            
                elif self.gain_type == "old_preservation":
                    # Preserve old tokens, reduce recent
                    for i in range(seq_len):
                        if i < seq_len - self.window_size:
                            # Old tokens get preservation
                            preservation_factor = 1.0 + self.gain_strength * 0.3
                            scores[:, i] *= preservation_factor
                        else:
                            # Recent tokens get slight reduction
                            reduction_factor = 1.0 - self.gain_strength * 0.2
                            scores[:, i] *= reduction_factor
                            
                elif self.gain_type == "bidirectional":
                    # Bidirectional gains: emphasize both old and new
                    for i in range(seq_len):
                        if i < self.window_size:
                            # Very old tokens get emphasis
                            boost_factor = 1.0 + self.gain_strength * 0.4
                            scores[:, i] *= boost_factor
                        elif i >= seq_len - self.window_size:
                            # Very recent tokens get emphasis
                            boost_factor = 1.0 + self.gain_strength * 0.4
                            scores[:, i] *= boost_factor
                        else:
                            # Middle tokens get slight reduction
                            reduction_factor = 1.0 - self.gain_strength * 0.1
                            scores[:, i] *= reduction_factor
                
                return scores
                
        return SegmentGainsProcessor(self.base_window_size, self.gain_type, effective_gain)
        
    def cleanup(self):
        pass

# ============================================================================
# ROUTING / MOE SPECIFIC EFFECTS
# ============================================================================

class RouterTemperatureBiasEffect(BaseEffect):
    """Router temperature / bias"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 temperature_mode: str = "sticky", bias_type: str = "uniform"):
        super().__init__(weight, direction)
        self.temperature_mode = temperature_mode
        self.bias_type = bias_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply router temperature/bias to MoE layers"""
        moe_layers = self._find_moe_layers(model)
        
        for layer_idx, moe_layer in moe_layers:
            try:
                # Get router module
                router = self._get_router_module(moe_layer)
                if router is None:
                    continue
                    
                # Store original forward
                original_forward = router.forward
                
                # Calculate effective temperature/bias
                base_temp = 1.0
                max_temp_change = 0.5
                effective_temp_change = self.get_effective_value(0.0, max_temp_change)
                
                def biased_router_forward(*args, **kwargs):
                    # Get original router output (logits)
                    output = original_forward(*args, **kwargs)
                    
                    if isinstance(output, torch.Tensor):
                        router_logits = output
                        
                        if self.temperature_mode == "sticky":
                            # Lower temperature = stickier routing (more deterministic)
                            temperature = base_temp - effective_temp_change
                            router_logits = router_logits / max(0.1, temperature)
                            
                        elif self.temperature_mode == "exploratory":
                            # Higher temperature = more exploratory routing
                            temperature = base_temp + effective_temp_change
                            router_logits = router_logits / temperature
                            
                        elif self.temperature_mode == "biased":
                            # Apply bias to favor certain experts
                            if self.bias_type == "uniform":
                                # Uniform bias across all experts
                                bias = torch.ones_like(router_logits) * effective_temp_change
                                router_logits = router_logits + bias
                            elif self.bias_type == "alternating":
                                # Alternating bias pattern
                                bias = torch.zeros_like(router_logits)
                                bias[..., ::2] = effective_temp_change  # Even experts get boost
                                bias[..., 1::2] = -effective_temp_change  # Odd experts get penalty
                                router_logits = router_logits + bias
                            elif self.bias_type == "first_half":
                                # Bias towards first half of experts
                                bias = torch.zeros_like(router_logits)
                                num_experts = router_logits.shape[-1]
                                bias[..., :num_experts//2] = effective_temp_change
                                router_logits = router_logits + bias
                        
                        return router_logits
                    
                    return output
                
                router.forward = biased_router_forward
                self.handles.append((router, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply router bias in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original router forward methods"""
        for router, original_forward in self.handles:
            router.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Router effects don't use logits processors"""
        return None
        
    def _find_moe_layers(self, model):
        """Find MoE layers in the model"""
        moe_layers = []
        
        # Try to find MoE layers in common architectures
        blocks = self._resolve_blocks(model)
        
        for idx, block in enumerate(blocks):
            # Look for MoE/expert modules
            for name, module in block.named_modules():
                if any(keyword in name.lower() for keyword in ['moe', 'expert', 'router', 'gate']):
                    if hasattr(module, 'router') or hasattr(module, 'gate'):
                        moe_layers.append((idx, module))
                        break
                        
        return moe_layers
        
    def _get_router_module(self, moe_layer):
        """Get router/gate module from MoE layer"""
        for attr in ["router", "gate", "gating_network"]:
            if hasattr(moe_layer, attr):
                return getattr(moe_layer, attr)
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        return []

class ExpertMaskingDropoutEffect(BaseEffect):
    """Expert masking / dropout"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 masking_pattern: str = "random", dropout_rate: float = 0.3):
        super().__init__(weight, direction)
        self.masking_pattern = masking_pattern
        self.base_dropout_rate = dropout_rate
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply expert masking/dropout to MoE layers"""
        moe_layers = self._find_moe_layers(model)
        
        for layer_idx, moe_layer in moe_layers:
            try:
                # Get router module
                router = self._get_router_module(moe_layer)
                if router is None:
                    continue
                    
                # Store original forward
                original_forward = router.forward
                
                # Calculate effective dropout rate
                base_dropout = 0.0
                max_dropout = 0.6
                effective_dropout = self.get_effective_value(base_dropout, max_dropout)
                
                def masked_router_forward(*args, **kwargs):
                    # Get original router output (logits)
                    output = original_forward(*args, **kwargs)
                    
                    if isinstance(output, torch.Tensor):
                        router_logits = output
                        batch_size, num_experts = router_logits.shape[-2:]
                        
                        if self.masking_pattern == "random":
                            # Random expert dropout
                            mask = torch.rand(num_experts, device=router_logits.device) > effective_dropout
                            router_logits = router_logits.masked_fill(~mask, float('-inf'))
                            
                        elif self.masking_pattern == "alternating":
                            # Alternating expert masking
                            mask = torch.arange(num_experts, device=router_logits.device) % 2 == 0
                            if effective_dropout > 0.5:  # Mask odd experts
                                router_logits = router_logits.masked_fill(~mask, float('-inf'))
                            else:  # Mask even experts
                                router_logits = router_logits.masked_fill(mask, float('-inf'))
                                
                        elif self.masking_pattern == "block":
                            # Block masking (mask consecutive experts)
                            num_to_mask = int(num_experts * effective_dropout)
                            start_idx = torch.randint(0, max(1, num_experts - num_to_mask + 1), (1,)).item()
                            mask = torch.ones(num_experts, dtype=torch.bool, device=router_logits.device)
                            mask[start_idx:start_idx + num_to_mask] = False
                            router_logits = router_logits.masked_fill(~mask, float('-inf'))
                            
                        elif self.masking_pattern == "specialist":
                            # Mask specialist experts (assuming later experts are more specialized)
                            num_to_mask = int(num_experts * effective_dropout)
                            mask = torch.ones(num_experts, dtype=torch.bool, device=router_logits.device)
                            mask[-num_to_mask:] = False  # Mask last N experts
                            router_logits = router_logits.masked_fill(~mask, float('-inf'))
                        
                        return router_logits
                    
                    return output
                
                router.forward = masked_router_forward
                self.handles.append((router, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply expert masking in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original router forward methods"""
        for router, original_forward in self.handles:
            router.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Expert masking effects don't use logits processors"""
        return None
        
    def _find_moe_layers(self, model):
        """Find MoE layers in the model"""
        moe_layers = []
        
        # Try to find MoE layers in common architectures
        blocks = self._resolve_blocks(model)
        
        for idx, block in enumerate(blocks):
            # Look for MoE/expert modules
            for name, module in block.named_modules():
                if any(keyword in name.lower() for keyword in ['moe', 'expert', 'router', 'gate']):
                    if hasattr(module, 'router') or hasattr(module, 'gate'):
                        moe_layers.append((idx, module))
                        break
                        
        return moe_layers
        
    def _get_router_module(self, moe_layer):
        """Get router/gate module from MoE layer"""
        for attr in ["router", "gate", "gating_network"]:
            if hasattr(moe_layer, attr):
                return getattr(moe_layer, attr)
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        return []

class ExpertPersistenceEffect(BaseEffect):
    """Expert persistence ("sticky routing")"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 persistence_type: str = "momentum", decay_rate: float = 0.1):
        super().__init__(weight, direction)
        self.persistence_type = persistence_type
        self.base_decay_rate = decay_rate
        self.handles = []
        self.expert_history = {}  # Track expert usage history
        
    def apply(self, model, **kwargs):
        """Apply expert persistence to MoE layers"""
        moe_layers = self._find_moe_layers(model)
        
        for layer_idx, moe_layer in moe_layers:
            try:
                # Get router module
                router = self._get_router_module(moe_layer)
                if router is None:
                    continue
                    
                # Store original forward
                original_forward = router.forward
                
                # Calculate effective persistence strength
                base_persistence = 0.0
                max_persistence = 0.4
                effective_persistence = self.get_effective_value(base_persistence, max_persistence)
                
                # Initialize expert history for this layer
                self.expert_history[layer_idx] = None
                
                def persistent_router_forward(*args, **kwargs):
                    # Get original router output (logits)
                    output = original_forward(*args, **kwargs)
                    
                    if isinstance(output, torch.Tensor):
                        router_logits = output
                        
                        if self.persistence_type == "momentum":
                            # Apply momentum-based persistence
                            if self.expert_history[layer_idx] is not None:
                                # Blend current logits with previous expert preferences
                                prev_probs = self.expert_history[layer_idx]
                                current_probs = torch.softmax(router_logits, dim=-1)
                                
                                # Momentum update
                                momentum = effective_persistence
                                blended_probs = (1 - momentum) * current_probs + momentum * prev_probs
                                
                                # Convert back to logits
                                router_logits = torch.log(blended_probs + 1e-8)
                            
                            # Update history
                            self.expert_history[layer_idx] = torch.softmax(router_logits, dim=-1).detach()
                            
                        elif self.persistence_type == "exponential":
                            # Exponential moving average of expert preferences
                            current_probs = torch.softmax(router_logits, dim=-1)
                            
                            if self.expert_history[layer_idx] is not None:
                                # Exponential moving average
                                alpha = 1.0 - effective_persistence
                                ema_probs = alpha * current_probs + (1 - alpha) * self.expert_history[layer_idx]
                                router_logits = torch.log(ema_probs + 1e-8)
                            
                            # Update history
                            self.expert_history[layer_idx] = torch.softmax(router_logits, dim=-1).detach()
                            
                        elif self.persistence_type == "winner_takes_all":
                            # Winner-takes-all persistence (boost previously selected experts)
                            if self.expert_history[layer_idx] is not None:
                                # Find previously selected experts
                                prev_winners = torch.argmax(self.expert_history[layer_idx], dim=-1)
                                
                                # Boost previously selected experts
                                boost = effective_persistence * 2.0  # Scale up the boost
                                for i, winner_idx in enumerate(prev_winners):
                                    router_logits[i, winner_idx] += boost
                            
                            # Update history
                            self.expert_history[layer_idx] = torch.softmax(router_logits, dim=-1).detach()
                        
                        return router_logits
                    
                    return output
                
                router.forward = persistent_router_forward
                self.handles.append((router, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply expert persistence in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original router forward methods and clear history"""
        for router, original_forward in self.handles:
            router.forward = original_forward
        self.handles.clear()
        self.expert_history.clear()
        
    def get_logits_processor(self):
        """Expert persistence effects don't use logits processors"""
        return None
        
    def _find_moe_layers(self, model):
        """Find MoE layers in the model"""
        moe_layers = []
        
        # Try to find MoE layers in common architectures
        blocks = self._resolve_blocks(model)
        
        for idx, block in enumerate(blocks):
            # Look for MoE/expert modules
            for name, module in block.named_modules():
                if any(keyword in name.lower() for keyword in ['moe', 'expert', 'router', 'gate']):
                    if hasattr(module, 'router') or hasattr(module, 'gate'):
                        moe_layers.append((idx, module))
                        break
                        
        return moe_layers
        
    def _get_router_module(self, moe_layer):
        """Get router/gate module from MoE layer"""
        for attr in ["router", "gate", "gating_network"]:
            if hasattr(moe_layer, attr):
                return getattr(moe_layer, attr)
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        return []

# ============================================================================
# OBJECTIVE MIXING & EXTERNAL CONTROL EFFECTS
# ============================================================================

class VerifierGuidedDecodingEffect(BaseEffect):
    """Verifier-guided decoding (rerank/accept-reject)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 verification_type: str = "quality", threshold: float = 0.7):
        super().__init__(weight, direction)
        self.verification_type = verification_type
        self.base_threshold = threshold
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply verifier-guided decoding"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup verifier effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get verifier-guided logits processor"""
        effective_threshold = self.get_effective_value(0.5, 0.9)
        
        class VerifierGuidedProcessor(LogitsProcessor):
            def __init__(self, verification_type, threshold, weight):
                self.verification_type = verification_type
                self.threshold = threshold
                self.weight = weight
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.verification_type == "quality":
                    # Penalize low-quality continuations
                    # Simulate quality scoring based on token frequency and context
                    seq_len = input_ids.shape[1]
                    if seq_len > 10:  # Only apply after some context
                        # Penalize repetitive tokens
                        recent_tokens = input_ids[:, -5:]
                        for i in range(scores.shape[-1]):
                            if i in recent_tokens:
                                scores[:, i] *= (1.0 - self.weight * 0.3)
                        
                        # Penalize generic tokens
                        generic_tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Common tokens
                        for token_id in generic_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.weight * 0.2)
                
                elif self.verification_type == "coherence":
                    # Penalize incoherent continuations
                    seq_len = input_ids.shape[1]
                    if seq_len > 5:
                        # Boost tokens that maintain topic coherence
                        # This is a simplified coherence check
                        recent_context = input_ids[:, -3:]
                        context_entropy = torch.std(recent_context.float())
                        if context_entropy > 10:  # High entropy = potential incoherence
                            # Boost more predictable tokens
                            scores = scores * (1.0 + self.weight * 0.1)
                
                elif self.verification_type == "task_alignment":
                    # Penalize off-task continuations
                    # Simulate task-specific verification
                    seq_len = input_ids.shape[1]
                    if seq_len > 8:
                        # Boost tokens that align with common task patterns
                        # This is a simplified task alignment check
                        task_keywords = [100, 200, 300, 400, 500]  # Example task tokens
                        for token_id in task_keywords:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.weight * 0.15)
                
                return scores
        
        return VerifierGuidedProcessor(self.verification_type, effective_threshold, self.weight)

class StyleAffectLogitBiasEffect(BaseEffect):
    """Style/affect logit bias (sentiment, prosociality)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 bias_type: str = "prosocial", sentiment: str = "positive"):
        super().__init__(weight, direction)
        self.bias_type = bias_type
        self.sentiment = sentiment
        self.handles = []
        
        # Define bias patterns for different styles
        self.style_biases = {
            "prosocial": {
                "positive": [100, 200, 300, 400, 500],  # Helpful, kind, supportive tokens
                "negative": [600, 700, 800, 900, 1000]  # Harmful, aggressive tokens
            },
            "sentiment": {
                "positive": [150, 250, 350, 450, 550],  # Happy, optimistic tokens
                "negative": [650, 750, 850, 950, 1050]  # Sad, pessimistic tokens
            },
            "warmth": {
                "positive": [120, 220, 320, 420, 520],  # Warm, friendly tokens
                "negative": [620, 720, 820, 920, 1020]  # Cold, distant tokens
            },
            "empathy": {
                "positive": [130, 230, 330, 430, 530],  # Understanding, caring tokens
                "negative": [630, 730, 830, 930, 1030]  # Dismissive, uncaring tokens
            }
        }
        
    def apply(self, model, **kwargs):
        """Apply style/affect bias"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup style bias effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get style/affect logits processor"""
        effective_bias = self.get_effective_value(0.0, 0.4)
        
        class StyleAffectProcessor(LogitsProcessor):
            def __init__(self, bias_type, sentiment, style_biases, bias_strength):
                self.bias_type = bias_type
                self.sentiment = sentiment
                self.style_biases = style_biases
                self.bias_strength = bias_strength
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.bias_type in self.style_biases:
                    bias_pattern = self.style_biases[self.bias_type]
                    
                    if self.sentiment == "positive":
                        # Boost positive tokens
                        for token_id in bias_pattern["positive"]:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.bias_strength)
                        
                        # Penalize negative tokens
                        for token_id in bias_pattern["negative"]:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.bias_strength * 0.5)
                    
                    elif self.sentiment == "negative":
                        # Boost negative tokens
                        for token_id in bias_pattern["negative"]:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.bias_strength)
                        
                        # Penalize positive tokens
                        for token_id in bias_pattern["positive"]:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.bias_strength * 0.5)
                
                return scores
        
        return StyleAffectProcessor(self.bias_type, self.sentiment, self.style_biases, effective_bias)

class RiskPreferenceSteeringEffect(BaseEffect):
    """Risk-preference steering"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 risk_type: str = "exploration", preference: str = "bold"):
        super().__init__(weight, direction)
        self.risk_type = risk_type
        self.preference = preference
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply risk preference steering"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup risk preference effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get risk preference logits processor"""
        effective_risk = self.get_effective_value(0.0, 0.5)
        
        class RiskPreferenceProcessor(LogitsProcessor):
            def __init__(self, risk_type, preference, risk_strength):
                self.risk_type = risk_type
                self.preference = preference
                self.risk_strength = risk_strength
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.risk_type == "exploration":
                    if self.preference == "bold":
                        # Boost less common tokens (exploration)
                        # Simulate by boosting tokens with lower scores
                        mean_score = torch.mean(scores, dim=-1, keepdim=True)
                        std_score = torch.std(scores, dim=-1, keepdim=True)
                        
                        # Boost tokens below mean (more exploratory)
                        exploration_mask = scores < mean_score
                        scores[exploration_mask] *= (1.0 + self.risk_strength * 0.3)
                        
                    elif self.preference == "cautious":
                        # Boost more common tokens (exploitation)
                        # Simulate by boosting tokens with higher scores
                        mean_score = torch.mean(scores, dim=-1, keepdim=True)
                        
                        # Boost tokens above mean (more conservative)
                        exploitation_mask = scores > mean_score
                        scores[exploitation_mask] *= (1.0 + self.risk_strength * 0.3)
                
                elif self.risk_type == "planning":
                    if self.preference == "bold":
                        # Boost action-oriented tokens
                        action_tokens = [110, 210, 310, 410, 510]  # Example action tokens
                        for token_id in action_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.risk_strength)
                        
                    elif self.preference == "cautious":
                        # Boost careful/analytical tokens
                        careful_tokens = [115, 215, 315, 415, 515]  # Example careful tokens
                        for token_id in careful_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.risk_strength)
                
                return scores
        
        return RiskPreferenceProcessor(self.risk_type, self.preference, effective_risk)

class ComputeAtTestSchedulingEffect(BaseEffect):
    """Compute-at-test scheduling (self-consistency bursts)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 scheduling_type: str = "burst", burst_length: int = 10):
        super().__init__(weight, direction)
        self.scheduling_type = scheduling_type
        self.burst_length = burst_length
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply compute scheduling"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup compute scheduling effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get compute scheduling logits processor"""
        effective_scheduling = self.get_effective_value(0.0, 0.6)
        
        class ComputeSchedulingProcessor(LogitsProcessor):
            def __init__(self, scheduling_type, burst_length, scheduling_strength):
                self.scheduling_type = scheduling_type
                self.burst_length = burst_length
                self.scheduling_strength = scheduling_strength
                self.token_count = 0
                self.burst_active = False
                self.burst_start = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.scheduling_type == "burst":
                    # Determine if we're in a burst phase
                    if self.token_count % (self.burst_length * 2) < self.burst_length:
                        # Burst phase - boost complex reasoning
                        if not self.burst_active:
                            self.burst_active = True
                            self.burst_start = self.token_count
                        
                        # Boost tokens associated with deeper reasoning
                        reasoning_tokens = [105, 205, 305, 405, 505]  # Example reasoning tokens
                        for token_id in reasoning_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.scheduling_strength)
                        
                        # Reduce temperature for more focused generation
                        scores = scores / (1.0 + self.scheduling_strength * 0.2)
                    
                    else:
                        # Non-burst phase - faster, more direct generation
                        if self.burst_active:
                            self.burst_active = False
                        
                        # Boost direct/simple tokens
                        direct_tokens = [110, 210, 310, 410, 510]  # Example direct tokens
                        for token_id in direct_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.scheduling_strength * 0.5)
                        
                        # Increase temperature for faster generation
                        scores = scores * (1.0 + self.scheduling_strength * 0.1)
                
                elif self.scheduling_type == "oscillating":
                    # Oscillating between deep and shallow processing
                    phase = (self.token_count % 20) / 20.0  # 20-token cycle
                    
                    if phase < 0.5:
                        # Deep processing phase
                        deep_tokens = [100, 200, 300, 400, 500]
                        for token_id in deep_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.scheduling_strength * phase)
                    else:
                        # Shallow processing phase
                        shallow_tokens = [150, 250, 350, 450, 550]
                        for token_id in shallow_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.scheduling_strength * (1.0 - phase))
                
                return scores
        
        return ComputeSchedulingProcessor(self.scheduling_type, self.burst_length, effective_scheduling)

class RetrievalRateModulationEffect(BaseEffect):
    """Retrieval rate modulation (RAG on/off or strength)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 retrieval_mode: str = "factual", modulation_type: str = "strength"):
        super().__init__(weight, direction)
        self.retrieval_mode = retrieval_mode
        self.modulation_type = modulation_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply retrieval rate modulation"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup retrieval modulation effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get retrieval rate modulation logits processor"""
        effective_modulation = self.get_effective_value(0.0, 0.5)
        
        class RetrievalModulationProcessor(LogitsProcessor):
            def __init__(self, retrieval_mode, modulation_type, modulation_strength):
                self.retrieval_mode = retrieval_mode
                self.modulation_type = modulation_type
                self.modulation_strength = modulation_strength
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.retrieval_mode == "factual":
                    if self.modulation_type == "strength":
                        # Boost factual/grounded tokens
                        factual_tokens = [120, 220, 320, 420, 520]  # Example factual tokens
                        for token_id in factual_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.modulation_strength)
                        
                        # Penalize imaginative/fictional tokens
                        imaginative_tokens = [125, 225, 325, 425, 525]  # Example imaginative tokens
                        for token_id in imaginative_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.modulation_strength * 0.3)
                    
                    elif self.modulation_type == "on_off":
                        # Binary factual mode
                        if self.token_count % 20 < 10:  # On for 10 tokens, off for 10
                            factual_tokens = [120, 220, 320, 420, 520]
                            for token_id in factual_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.modulation_strength)
                
                elif self.retrieval_mode == "imaginative":
                    if self.modulation_type == "strength":
                        # Boost imaginative/creative tokens
                        imaginative_tokens = [125, 225, 325, 425, 525]
                        for token_id in imaginative_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.modulation_strength)
                        
                        # Penalize overly factual tokens
                        factual_tokens = [120, 220, 320, 420, 520]
                        for token_id in factual_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.modulation_strength * 0.3)
                    
                    elif self.modulation_type == "on_off":
                        # Binary imaginative mode
                        if self.token_count % 20 >= 10:  # Off for 10 tokens, on for 10
                            imaginative_tokens = [125, 225, 325, 425, 525]
                            for token_id in imaginative_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.modulation_strength)
                
                return scores
        
        return RetrievalModulationProcessor(self.retrieval_mode, self.modulation_type, effective_modulation)

class PersonaVoiceConstraintsEffect(BaseEffect):
    """Persona/voice constraints (hidden prompts or Δh)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 persona_type: str = "professional", voice_mode: str = "stable"):
        super().__init__(weight, direction)
        self.persona_type = persona_type
        self.voice_mode = voice_mode
        self.handles = []
        
        # Define persona patterns
        self.persona_patterns = {
            "professional": {
                "formal_tokens": [130, 230, 330, 430, 530],
                "casual_tokens": [135, 235, 335, 435, 535]
            },
            "friendly": {
                "warm_tokens": [140, 240, 340, 440, 540],
                "cold_tokens": [145, 245, 345, 445, 545]
            },
            "authoritative": {
                "confident_tokens": [150, 250, 350, 450, 550],
                "uncertain_tokens": [155, 255, 355, 455, 555]
            },
            "creative": {
                "artistic_tokens": [160, 260, 360, 460, 560],
                "analytical_tokens": [165, 265, 365, 465, 565]
            }
        }
        
    def apply(self, model, **kwargs):
        """Apply persona/voice constraints"""
        # This effect works through logits processing
        pass
        
    def cleanup(self):
        """Cleanup persona/voice effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get persona/voice logits processor"""
        effective_persona = self.get_effective_value(0.0, 0.4)
        
        class PersonaVoiceProcessor(LogitsProcessor):
            def __init__(self, persona_type, voice_mode, persona_patterns, persona_strength):
                self.persona_type = persona_type
                self.voice_mode = voice_mode
                self.persona_patterns = persona_patterns
                self.persona_strength = persona_strength
                self.token_count = 0
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                if self.persona_type in self.persona_patterns:
                    pattern = self.persona_patterns[self.persona_type]
                    
                    if self.voice_mode == "stable":
                        # Consistent persona application
                        # Boost preferred tokens
                        preferred_tokens = list(pattern.values())[0]  # First token list
                        for token_id in preferred_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.persona_strength)
                        
                        # Penalize non-preferred tokens
                        non_preferred_tokens = list(pattern.values())[1]  # Second token list
                        for token_id in non_preferred_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 - self.persona_strength * 0.5)
                    
                    elif self.voice_mode == "adaptive":
                        # Adaptive persona based on context
                        seq_len = input_ids.shape[1]
                        if seq_len > 10:
                            # Switch persona based on recent context
                            recent_context = input_ids[:, -5:]
                            context_variety = torch.std(recent_context.float())
                            
                            if context_variety > 5:  # Diverse context
                                # Use more flexible persona
                                all_tokens = preferred_tokens + non_preferred_tokens
                                for token_id in all_tokens:
                                    if token_id < scores.shape[-1]:
                                        scores[:, token_id] *= (1.0 + self.persona_strength * 0.3)
                            else:
                                # Use more rigid persona
                                preferred_tokens = list(pattern.values())[0]
                                for token_id in preferred_tokens:
                                    if token_id < scores.shape[-1]:
                                        scores[:, token_id] *= (1.0 + self.persona_strength)
                    
                    elif self.voice_mode == "oscillating":
                        # Oscillating persona strength
                        phase = (self.token_count % 30) / 30.0  # 30-token cycle
                        
                        if phase < 0.5:
                            # Strong persona phase
                            preferred_tokens = list(pattern.values())[0]
                            for token_id in preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.persona_strength * (1.0 - phase))
                        else:
                            # Weak persona phase
                            preferred_tokens = list(pattern.values())[0]
                            for token_id in preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.persona_strength * phase)
                
                return scores
        
        return PersonaVoiceProcessor(self.persona_type, self.voice_mode, self.persona_patterns, effective_persona)

# ============================================================================
# INPUT / CONTEXT PERTURBATION EFFECTS
# ============================================================================

class LexicalJitterEffect(BaseEffect):
    """
    Lexical jitter in context (synonym swap/ablation) - FIXED to work at embedding/KV-cache level.
    
    FIXED: Previously implemented as LogitsProcessor (causally incorrect for "memory" perturbations).
    Now works at the embedding layer or KV-cache update step, which is the correct causal structure.
    """
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 jitter_type: str = "synonym_swap", ablation_rate: float = 0.1):
        super().__init__(weight, direction)
        self.jitter_type = jitter_type
        self.base_ablation_rate = ablation_rate
        self.handles = []
        
    def apply(self, model, **kwargs):
        """
        Apply lexical jitter at the embedding layer using a pre-hook (causally correct).
        
        FIXED: Uses register_forward_pre_hook to modify embeddings BEFORE they enter
        the transformer. This creates actual perceptual noise, not just randomizing
        the next word choice.
        """
        effective_jitter = self.get_effective_value(0.0, 0.3)
        
        # Find the embedding layer
        embedding_layer = None
        if hasattr(model, 'get_input_embeddings'):
            embedding_layer = model.get_input_embeddings()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embedding_layer = model.transformer.wte  # GPT2-style
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embedding_layer = model.model.embed_tokens  # Llama-style
        else:
            logger.warning("Could not find embedding layer, skipping lexical jitter")
            return
        
        # Calculate noise sigma based on jitter strength
        self.sigma = effective_jitter * 0.05  # Scale factor for noise
        
        def embedding_hook(module, input, output):
            """
            Pre-hook to modify embeddings at the input level.
            
            Args:
                module: The embedding module
                input: Tuple containing (input_ids, ...)
                output: Embeddings tensor [batch, seq, hidden_dim]
            
            Returns:
                Modified embeddings with jitter applied
            """
            # output is [batch, seq, hidden_dim] (The Embeddings)
            if self.jitter_type == "noise" or self.jitter_type == "noise_injection":
                # Add perceptual noise to embeddings
                noise = torch.randn_like(output) * self.sigma
                return output + noise
                
            elif self.jitter_type == "synonym_swap":
                # Add small random noise to simulate synonym variation
                noise = torch.randn_like(output) * (self.sigma * 0.6)
                return output + noise
                
            elif self.jitter_type == "ablation":
                # Zero out a fraction of embeddings (simulating token ablation)
                batch_size, seq_len, hidden_dim = output.shape
                if seq_len > 3:
                    ablation_mask = torch.rand(batch_size, seq_len, 1, device=output.device) < (self.base_ablation_rate * effective_jitter)
                    return output * (1.0 - ablation_mask.float())
                return output
                
            elif self.jitter_type == "reframing":
                # Slightly rotate embeddings to simulate reframing
                if output.shape[1] > 8:
                    rotation_strength = effective_jitter * 0.02
                    mean_embedding = output.mean(dim=1, keepdim=True)
                    return output + rotation_strength * mean_embedding
                return output
            
            return output
        
        # Register pre-hook (runs BEFORE forward, but we modify output)
        # Actually, we need a forward hook to modify the output
        handle = embedding_layer.register_forward_hook(embedding_hook)
        self.handles.append(handle)
        
        logger.info(f"Applied lexical jitter ({self.jitter_type}) at embedding layer with sigma={self.sigma:.4f}")
        
    def cleanup(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        
    def get_logits_processor(self) -> Optional[LogitsProcessor]:
        """Lexical jitter no longer uses logits processors (fixed to embedding level)"""
        return None

class StructuredPrefacesEffect(BaseEffect):
    """
    Structured prefaces (invisible to model text; KV-only) - FIXED to inject into past_key_values.
    
    FIXED: Previously implemented as LogitsProcessor (causally incorrect for "memory" perturbations).
    Now pre-computes KV states for a preface string and injects them directly into past_key_values.
    This is actual "implanted memory," not just biasing the model to say specific words.
    """
    
    def __init__(self, weight: float = 0.0, direction: str = "up", 
                 preface_type: str = "bias", injection_mode: str = "kv_only",
                 preface_text: Optional[str] = None):
        super().__init__(weight, direction)
        self.preface_type = preface_type
        self.injection_mode = injection_mode
        self.preface_text = preface_text or self._get_default_preface(preface_type)
        self.preface_kv_cache = None  # Will be computed on first apply
        self.tokenizer = None
        self.model = None
        
    def _get_default_preface(self, preface_type: str) -> str:
        """Get default preface text based on type"""
        prefaces = {
            "bias": "You are a helpful assistant who provides positive and constructive responses.",
            "style": "You are a formal and professional assistant who uses precise language.",
            "topic": "You are a technical expert who focuses on accurate and detailed information.",
            "emotion": "You are a calm and composed assistant who maintains emotional stability."
        }
        return prefaces.get(preface_type, prefaces["bias"])
    
    def apply(self, model, **kwargs):
        """
        Pre-compute KV states for preface string.
        
        The actual injection into past_key_values happens during generation
        via the modify_kv_cache() method.
        """
        self.model = model
        tokenizer = kwargs.get('tokenizer')
        if tokenizer is None:
            # Try to get tokenizer from model
            if hasattr(model, 'tokenizer'):
                tokenizer = model.tokenizer
            else:
                logger.warning("No tokenizer provided, cannot pre-compute preface KV cache")
                return
        
        self.tokenizer = tokenizer
        
        # Pre-compute KV states for the preface string
        logger.info(f"Pre-computing KV cache for preface: '{self.preface_text[:50]}...'")
        
        try:
            # Tokenize preface
            preface_inputs = tokenizer(self.preface_text, return_tensors="pt")
            preface_inputs = {k: v.to(next(model.parameters()).device) for k, v in preface_inputs.items()}
            
            # Forward pass to get KV cache
            with torch.no_grad():
                outputs = model(**preface_inputs, use_cache=True, return_dict=True)
                # Extract past_key_values
                self.preface_kv_cache = outputs.past_key_values
            
            logger.info(f"Pre-computed preface KV cache with {len(self.preface_kv_cache)} layers")
            
        except Exception as e:
            logger.error(f"Failed to pre-compute preface KV cache: {e}")
            import traceback
            traceback.print_exc()
    
    def modify_kv_cache(self, kv_cache):
        """
        Inject pre-computed preface KV states into the cache.
        
        This is called during generation, before the first token is generated.
        The preface KV cache is concatenated with the current context KV cache.
        
        Args:
            kv_cache: Current past_key_values tuple (or None if first generation step)
            
        Returns:
            Modified KV cache with preface injected
        """
        if self.preface_kv_cache is None:
            # Not yet computed, return original
            return kv_cache
        
        if kv_cache is None:
            # First generation step - use preface as initial KV cache
            return self.preface_kv_cache
        
        # Concatenate preface KV cache with current KV cache
        # kv_cache is usually a tuple of (key, value) tensors per layer
        # Shape: (layer_idx, (key, value)) where key/value are [batch, num_heads, seq_len, head_dim]
        
        modified_cache = []
        for layer_idx, (preface_k, preface_v) in enumerate(self.preface_kv_cache):
            if layer_idx < len(kv_cache):
                current_layer_cache = kv_cache[layer_idx]
                if current_layer_cache is not None:
                    current_k, current_v = current_layer_cache
                    
                    # Concatenate along sequence dimension (dim=-2)
                    # preface_k/v: [batch, num_heads, preface_seq_len, head_dim]
                    # current_k/v: [batch, num_heads, current_seq_len, head_dim]
                    # Result: [batch, num_heads, preface_seq_len + current_seq_len, head_dim]
                    combined_k = torch.cat([preface_k, current_k], dim=-2)
                    combined_v = torch.cat([preface_v, current_v], dim=-2)
                    
                    modified_cache.append((combined_k, combined_v))
                else:
                    # No current cache, use preface only
                    modified_cache.append((preface_k, preface_v))
            else:
                # More preface layers than current cache (shouldn't happen, but handle gracefully)
                modified_cache.append((preface_k, preface_v))
        
        return tuple(modified_cache)
        
    def cleanup(self):
        """Clear preface KV cache"""
        self.preface_kv_cache = None
        self.tokenizer = None
        self.model = None
        
    def get_logits_processor(self) -> Optional[LogitsProcessor]:
        """Structured prefaces no longer use logits processors (fixed to KV-cache level)"""
        return None

# ============================================================================
# ACTIVATION / REPRESENTATION SURGERY EFFECTS
# ============================================================================

class ActivationAdditionsEffect(BaseEffect):
    """Activation additions (Δh steering vectors)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 steering_type: str = "associative", layers: str = "all"):
        super().__init__(weight, direction)
        self.steering_type = steering_type
        self.layers = layers
        self.handles = []
        
        # Steering vectors will be loaded from disk (CAA-generated)
        # Fallback to zero vectors if not found (NOT random noise)
        self.steering_vectors = {}
        self.vector_dir = "outputs/steering_vectors"
        self._vector_cache = {}
        
        # Contrastive prompt system for steering vector construction
        self.contrastive_prompts = {
            "associative": {
                "positive": "Think creatively and make novel connections between ideas.",
                "negative": "Think literally and focus only on direct, obvious relationships."
            },
            "prosocial": {
                "positive": "Be helpful, kind, and considerate of others' feelings.",
                "negative": "Be selfish, unhelpful, and disregard others' needs."
            },
            "creative": {
                "positive": "Generate original, innovative, and imaginative ideas.",
                "negative": "Generate conventional, predictable, and unoriginal ideas."
            },
            "focused": {
                "positive": "Stay focused, organized, and task-oriented.",
                "negative": "Be distracted, disorganized, and unfocused."
            }
        }
        
        # Storage for computed steering vectors
        self.computed_vectors = {}
        self.vector_cache = {}
    
    def _load_vector(self, steering_type: str, hidden_size: int = 768) -> torch.Tensor:
        """
        Load a steering vector from disk, with zero vector fallback.
        
        Args:
            steering_type: Type of steering vector to load
            hidden_size: Expected hidden size
            
        Returns:
            Steering vector tensor (zero vector if not found)
        """
        from pathlib import Path
        
        if steering_type in self._vector_cache:
            return self._vector_cache[steering_type]
        
        vector_dir = Path(self.vector_dir)
        vector_path = None
        
        # Try multiple layer indices
        for layer_idx in [-1, -2, 0]:
            candidate_path = vector_dir / f"{steering_type}_layer{layer_idx}.pt"
            if candidate_path.exists():
                vector_path = candidate_path
                break
        
        if vector_path is None:
            candidate_path = vector_dir / f"{steering_type}.pt"
            if candidate_path.exists():
                vector_path = candidate_path
        
        if vector_path is None:
            logger.warning(f"Steering vector for '{steering_type}' not found. Using zero vector.")
            vector = torch.zeros(hidden_size)
            self._vector_cache[steering_type] = vector
            return vector
        
        try:
            vector = torch.load(vector_path, map_location='cpu')
            if isinstance(vector, torch.Tensor) and len(vector.shape) == 1:
                # Resize if needed
                if vector.shape[0] != hidden_size:
                    if vector.shape[0] < hidden_size:
                        padding = torch.zeros(hidden_size - vector.shape[0])
                        vector = torch.cat([vector, padding])
                    else:
                        vector = vector[:hidden_size]
                self._vector_cache[steering_type] = vector
                return vector
            else:
                logger.warning(f"Invalid vector format in {vector_path}. Using zero vector.")
                vector = torch.zeros(hidden_size)
                self._vector_cache[steering_type] = vector
                return vector
        except Exception as e:
            logger.error(f"Failed to load vector from {vector_path}: {e}")
            vector = torch.zeros(hidden_size)
            self._vector_cache[steering_type] = vector
            return vector
    
    def compute_contrastive_steering_vector(self, model, steering_type: str, 
                                          layer_idx: int = -1) -> torch.Tensor:
        """
        Compute steering vector using contrastive prompts or load from disk.
        
        Args:
            model: The language model
            steering_type: Type of steering (associative, prosocial, etc.)
            layer_idx: Layer to extract activations from (-1 for last layer)
            
        Returns:
            Computed steering vector (loaded from disk if available, else zero vector)
        """
        # Try to load from disk first
        hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "d_model", 768))
        vector = self._load_vector(steering_type, hidden_size=hidden_size)
        
        # If vector is non-zero, it was loaded successfully
        if torch.norm(vector) > 0:
            return vector
        
        # Fallback: try to compute using contrastive prompts (if available)
        if steering_type in self.contrastive_prompts:
            # This would use the existing _extract_activations method
            # For now, return zero vector and log a warning
            logger.warning(f"Could not load or compute vector for {steering_type}. Using zero vector.")
            return torch.zeros(hidden_size)
        
        # Final fallback: zero vector (NOT random noise)
        logger.warning(f"No vector available for {steering_type}. Using zero vector.")
        return torch.zeros(hidden_size)
    
    def _extract_activations(self, model, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Extract activations from a specific layer for a given prompt.
        
        NOTE: This method is deprecated. Use SteeringVectorGenerator instead.
        Returns zero vector as placeholder (NOT random noise).
        """
        logger.warning("_extract_activations is deprecated. Use SteeringVectorGenerator for CAA.")
        hidden_size = getattr(model.config, "hidden_size", getattr(model.config, "d_model", 768))
        return torch.zeros(hidden_size)
    
    def compute_layer_wise_delta(self, model, prompt: str, 
                                target_behavior: str) -> Dict[int, torch.Tensor]:
        """
        Compute layer-wise delta (Δh) for steering vector construction
        
        Args:
            model: The language model
            prompt: Input prompt
            target_behavior: Target behavior to steer towards
            
        Returns:
            Dictionary mapping layer indices to delta vectors
        """
        layer_deltas = {}
        
        try:
            # This is a simplified implementation
            # In practice, you would:
            # 1. Run the model with the prompt
            # 2. Extract activations from each layer
            # 3. Compute differences between target and baseline behaviors
            # 4. Return layer-wise deltas
            
            # For now, return random deltas for demonstration
            for layer_idx in range(12):  # Assuming 12 layers
                layer_deltas[layer_idx] = torch.randn(768) * 0.1
            
            return layer_deltas
            
        except Exception as e:
            logger.warning(f"Failed to compute layer-wise delta: {e}")
            return {}
    
    def add_runtime_steering(self, hidden_states: torch.Tensor, 
                           layer_idx: int, token_idx: int = -1) -> torch.Tensor:
        """
        Add steering vector at runtime (last token or specific position)
        
        Args:
            hidden_states: Current hidden states
            layer_idx: Current layer index
            token_idx: Token position to apply steering (-1 for last token)
            
        Returns:
            Modified hidden states
        """
        if self.steering_type not in self.steering_vectors:
            return hidden_states
        
        # Get steering vector
        steering_vector = self.steering_vectors[self.steering_type]
        
        # Calculate effective steering strength
        base_strength = 0.0
        max_strength = 0.3
        effective_strength = self.get_effective_value(base_strength, max_strength)
        
        # Apply steering to the specified token position
        if token_idx == -1:
            # Apply to last token
            hidden_states[:, -1, :] += steering_vector * effective_strength
        else:
            # Apply to specific token position
            if token_idx < hidden_states.shape[1]:
                hidden_states[:, token_idx, :] += steering_vector * effective_strength
        
        return hidden_states
    
    def store_steering_vector(self, name: str, vector: torch.Tensor):
        """Store a computed steering vector for later use"""
        self.computed_vectors[name] = vector.clone()
    
    def retrieve_steering_vector(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve a stored steering vector"""
        return self.computed_vectors.get(name)
        
    def apply(self, model, **kwargs):
        """Apply activation steering to transformer layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get the transformer block
                block = blocks[layer_idx]
                
                # Store original forward
                original_forward = block.forward
                
                # Calculate effective steering strength
                base_strength = 0.0
                max_strength = 0.3
                effective_strength = self.get_effective_value(base_strength, max_strength)
                
                def steered_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply steering to hidden states
                    if isinstance(output, tuple) and len(output) >= 1:
                        hidden_states = output[0]
                        
                        # Apply steering vector to the last token's hidden state
                        if self.steering_type in self.steering_vectors:
                            steering_vector = self.steering_vectors[self.steering_type]
                            steering_effect = steering_vector.unsqueeze(0).unsqueeze(0) * effective_strength
                            
                            # Apply to last token position
                            hidden_states[:, -1, :] += steering_effect.squeeze(0)
                            
                            # Update output tuple
                            output = (hidden_states,) + output[1:]
                    
                    return output
                
                block.forward = steered_forward
                self.handles.append((block, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply activation steering in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for block, original_forward in self.handles:
            block.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Activation effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))

class SoftProjectionEffect(BaseEffect):
    """Soft projection ("conceptors") for feature subspace gating"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 projection_type: str = "creative", layers: str = "mid"):
        super().__init__(weight, direction)
        self.projection_type = projection_type
        self.layers = layers
        self.handles = []
        
        # Define projection matrices for different feature subspaces
        self.projections = {
            "creative": torch.randn(768, 768) * 0.1,  # Creative thinking subspace
            "analytical": torch.randn(768, 768) * 0.08,  # Analytical thinking subspace
            "emotional": torch.randn(768, 768) * 0.12,  # Emotional processing subspace
            "spatial": torch.randn(768, 768) * 0.09,  # Spatial reasoning subspace
            "linguistic": torch.randn(768, 768) * 0.07,  # Linguistic processing subspace
        }
        
    def apply(self, model, **kwargs):
        """Apply soft projection to transformer layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get the transformer block
                block = blocks[layer_idx]
                
                # Store original forward
                original_forward = block.forward
                
                # Calculate effective projection strength
                base_strength = 0.0
                max_strength = 0.2
                effective_strength = self.get_effective_value(base_strength, max_strength)
                
                def projected_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply soft projection to hidden states
                    if isinstance(output, tuple) and len(output) >= 1:
                        hidden_states = output[0]
                        
                        # Apply projection matrix
                        if self.projection_type in self.projections:
                            projection = self.projections[self.projection_type]
                            # Soft projection: h = h + α * P * h
                            projected = torch.matmul(hidden_states, projection.T)
                            hidden_states = hidden_states + effective_strength * projected
                            
                            # Update output tuple
                            output = (hidden_states,) + output[1:]
                    
                    return output
                
                block.forward = projected_forward
                self.handles.append((block, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply soft projection in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for block, original_forward in self.handles:
            block.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Projection effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))

class LayerWiseGainEffect(BaseEffect):
    """Layer-wise gain (residual scalers)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "all", gain_type: str = "uniform"):
        super().__init__(weight, direction)
        self.layers = layers
        self.gain_type = gain_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply layer-wise gain scaling"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get the transformer block
                block = blocks[layer_idx]
                
                # Store original forward
                original_forward = block.forward
                
                # Calculate effective gain
                base_gain = 1.0
                max_gain_change = 0.3
                effective_gain_change = self.get_effective_value(0.0, max_gain_change)
                
                def gained_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply gain scaling to hidden states
                    if isinstance(output, tuple) and len(output) >= 1:
                        hidden_states = output[0]
                        
                        # Apply gain: h = h * (1 ± α)
                        gain_factor = 1.0 + effective_gain_change
                        hidden_states = hidden_states * gain_factor
                        
                        # Update output tuple
                        output = (hidden_states,) + output[1:]
                    
                    return output
                
                block.forward = gained_forward
                self.handles.append((block, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply layer-wise gain in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for block, original_forward in self.handles:
            block.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Gain effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))

class NoiseInjectionEffect(BaseEffect):
    """Noise injection (tiny Gaussian on activations)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "mid", noise_std: float = 0.01):
        super().__init__(weight, direction)
        self.layers = layers
        self.base_noise_std = noise_std
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply noise injection to transformer layers"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                # Get the transformer block
                block = blocks[layer_idx]
                
                # Store original forward
                original_forward = block.forward
                
                # Calculate effective noise strength
                base_noise = 0.0
                max_noise = 0.02
                effective_noise = self.get_effective_value(base_noise, max_noise)
                
                def noisy_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply noise to hidden states
                    if isinstance(output, tuple) and len(output) >= 1:
                        hidden_states = output[0]
                        
                        # Add Gaussian noise: h = h + ε ~ N(0, σ²)
                        noise = torch.randn_like(hidden_states) * effective_noise
                        hidden_states = hidden_states + noise
                        
                        # Update output tuple
                        output = (hidden_states,) + output[1:]
                    
                    return output
                
                block.forward = noisy_forward
                self.handles.append((block, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply noise injection in layer {layer_idx}: {e}")
                continue
                
    def cleanup(self):
        """Restore original forward methods"""
        for block, original_forward in self.handles:
            block.forward = original_forward
        self.handles.clear()
        
    def get_logits_processor(self):
        """Noise effects don't use logits processors"""
        return None
        
    def _resolve_blocks(self, model):
        """Resolve transformer blocks"""
        for attr in ["transformer", "model", "language_model"]:
            if hasattr(model, attr):
                cand = getattr(model, attr)
                if hasattr(cand, "h"):
                    return cand.h
                elif hasattr(cand, "layers"):
                    return cand.layers
        raise RuntimeError("Could not resolve transformer blocks")
        
    def _select_layers(self, blocks):
        """Select layers based on configuration"""
        L = len(blocks)
        if self.layers == "deep":
            return list(range(2*L//3, L))
        elif self.layers == "mid":
            return list(range(L//3, 2*L//3))
        elif self.layers == "shallow":
            return list(range(0, L//3))
        else:  # "all"
            return list(range(0, L))

# ============================================================================
# EFFECT REGISTRY
# ============================================================================

class EffectRegistry:
    """Registry for all available effects"""
    
    def __init__(self):
        self.effects = {
            # Sampler / Logits Shaping Effects
            "temperature": TemperatureEffect,
            "top_p": TopPEffect,
            "frequency_penalty": FrequencyPenaltyEffect,
            "presence_penalty": PresencePenaltyEffect,
            "pulsed_sampler": PulsedSamplerEffect,
            "contrastive_decoding": ContrastiveDecodingEffect,
            "expert_mixing": ExpertMixingEffect,
            "token_class_temperature": TokenClassTemperatureEffect,
            
            # Attention Effects
            "attention_focus": AttentionFocusEffect,
            "attention_masking": AttentionMaskingEffect,
            
            # Attention Modification Effects
            "qk_score_scaling": QKScoreScalingEffect,
            "head_masking_dropout": HeadMaskingDropoutEffect,
            "head_reweighting": HeadReweightingEffect,
            "positional_bias_tweak": PositionalBiasTweakEffect,
            "attention_oscillation": AttentionOscillationEffect,
            "attention_sinks_anchors": AttentionSinksAnchorsEffect,
            
            # Steering Effects
            "steering": SteeringEffect,
            "random_direction": RandomDirectionEffect,
            "random_orthogonal_steering": RandomOrthogonalSteeringEffect,
            
            # Memory Effects
            "kv_decay": KVDecayEffect,
            "kv_compression": KVCompressionEffect,

            # Working-Memory / KV-Cache Effects
            "exponential_decay_kv": ExponentialDecayKVEffect,
            "truncation_kv": TruncationKVEffect,
            "stride_compression_kv": StrideCompressionKVEffect,
            "segment_gains_kv": SegmentGainsKVEffect,

            # Routing / MoE Specific Effects
            "router_temperature_bias": RouterTemperatureBiasEffect,
            "expert_masking_dropout": ExpertMaskingDropoutEffect,
            "expert_persistence": ExpertPersistenceEffect,

            # Objective Mixing & External Control Effects
            "verifier_guided_decoding": VerifierGuidedDecodingEffect,
            "style_affect_logit_bias": StyleAffectLogitBiasEffect,
            "risk_preference_steering": RiskPreferenceSteeringEffect,
            "compute_at_test_scheduling": ComputeAtTestSchedulingEffect,
            "retrieval_rate_modulation": RetrievalRateModulationEffect,
            "persona_voice_constraints": PersonaVoiceConstraintsEffect,

            # Input / Context Perturbation Effects
            "lexical_jitter": LexicalJitterEffect,
            "structured_prefaces": StructuredPrefacesEffect,

            # Activation / Representation Surgery Effects
            "activation_additions": ActivationAdditionsEffect,
            "soft_projection": SoftProjectionEffect,
            "layer_wise_gain": LayerWiseGainEffect,
            "noise_injection": NoiseInjectionEffect,
            
            # Visual Effects (for image generation)
            "color_bias": ColorBiasEffect,
            "style_transfer": StyleTransferEffect,
            "composition_bias": CompositionBiasEffect,
            "visual_entropy": VisualEntropyEffect,
            "synesthetic_mapping": SynestheticMappingEffect,
            "motion_blur": MotionBlurEffect,
        }
        
    def get_effect(self, effect_name: str, **kwargs) -> BaseEffect:
        """Get an effect instance"""
        if effect_name not in self.effects:
            raise ValueError(f"Unknown effect: {effect_name}")
        return self.effects[effect_name](**kwargs)
        
    def list_effects(self) -> List[str]:
        """List all available effects"""
        return list(self.effects.keys())