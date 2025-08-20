"""
Modular neuromodulation effects system
Each effect can be applied with weight (0.0-1.0) and direction (up/down/neutral)
"""

import torch
import torch.nn.functional as F
import random
import math
import contextlib
from typing import Dict, Any, List, Optional, Callable, Iterable, Union
from transformers import LogitsProcessor
import numpy as np
from abc import ABC, abstractmethod

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
                torch_dtype=torch.float32, 
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
                        # Get small model logits
                        small_inputs = self.small_tokenizer(
                            self.small_tokenizer.decode(input_ids[0]), 
                            return_tensors="pt", 
                            truncation=True, 
                            max_length=input_ids.shape[1]
                        )
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
    """Attention focus enhancement effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", layers: str = "mid"):
        super().__init__(weight, direction)
        self.layers = layers
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply QK scaling to attention"""
        blocks = self._resolve_blocks(model)
        selected_layers = self._select_layers(blocks)
        
        for layer_idx in selected_layers:
            try:
                attn = self._get_attention_module(blocks[layer_idx])
                if attn is None:
                    continue
                    
                # Calculate effective QK scale
                base_scale = 1.0
                max_scale = 0.3
                effective_scale = self.get_effective_value(base_scale, max_scale)
                
                # Store original forward
                original_forward = attn.forward
                
                def scaled_forward(*args, **kwargs):
                    output = original_forward(*args, **kwargs)
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            # Apply QK scaling
                            attn_weights = attn_weights * effective_scale
                            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                            
                            # Recompute attention output if possible
                            if len(output) >= 3 and output[2] is not None:
                                V = output[2]
                                new_attn_output = torch.matmul(attn_weights, V)
                                output = (new_attn_output,) + output[1:]
                    return output
                
                attn.forward = scaled_forward
                self.handles.append((attn, original_forward))
                    
            except Exception as e:
                print(f"Warning: Failed to apply attention focus in layer {layer_idx}: {e}")
                continue
                
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
    """QK score scaling (attention sharpness)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 layers: str = "mid", scaling_type: str = "uniform"):
        super().__init__(weight, direction)
        self.layers = layers
        self.scaling_type = scaling_type
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply QK score scaling to attention layers"""
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
                
                # Calculate effective scaling factor
                base_scale = 1.0
                max_scale_change = 0.3
                effective_scale_change = self.get_effective_value(0.0, max_scale_change)
                
                def scaled_forward(*args, **kwargs):
                    # Get original output
                    output = original_forward(*args, **kwargs)
                    
                    # Apply QK scaling to attention weights
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_weights = output[1]
                        if attn_weights is not None:
                            # Scale attention weights: attn = attn * (1 + δ)
                            scale_factor = 1.0 + effective_scale_change
                            attn_weights = attn_weights * scale_factor
                            
                            # Renormalize
                            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
                            
                            # Recompute attention output if possible
                            if len(output) >= 3 and output[2] is not None:
                                V = output[2]
                                new_attn_output = torch.matmul(attn_weights, V)
                                output = (new_attn_output,) + output[1:]
                    
                    return output
                
                attn.forward = scaled_forward
                self.handles.append((attn, original_forward))
                
            except Exception as e:
                print(f"Warning: Failed to apply QK scaling in layer {layer_idx}: {e}")
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
                                head_mask = torch.rand(num_heads) > effective_dropout
                                head_mask = head_mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                                attn_weights = attn_weights * head_mask
                            elif self.dropout_type == "alternating":
                                # Alternating head dropout
                                head_mask = torch.arange(num_heads) % 2 == 0
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
                                oscillation = torch.sin(torch.arange(seq_len) * 0.1) * effective_amplitude
                                oscillation = oscillation.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
                                attn_weights = attn_weights * (1.0 + oscillation)
                                
                            elif self.oscillation_type == "square":
                                # Square wave oscillation
                                oscillation = (torch.arange(seq_len) % 20 < 10).float() * effective_amplitude
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
    """Activation steering effect"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", steering_type: str = "associative"):
        super().__init__(weight, direction)
        self.steering_type = steering_type
        self.steering_vectors = {
            "associative": torch.randn(768) * 0.15,
            "visionary": torch.randn(768) * 0.18,
            "synesthesia": torch.randn(768) * 0.16,
            "ego_thin": torch.randn(768) * 0.14,
            "prosocial": torch.randn(768) * 0.12,
            "affiliative": torch.randn(768) * 0.10,
            "goal_focused": torch.randn(768) * 0.08,
            "playful": torch.randn(768) * 0.20,
            "creative": torch.randn(768) * 0.1,
            "abstract": torch.randn(768) * 0.12,
        }
        
    def apply(self, model, **kwargs):
        """Apply steering to hidden states"""
        # This will be applied during generation
        pass
        
    def apply_steering(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply steering to hidden states"""
        if self.steering_type in self.steering_vectors:
            steering_vector = self.steering_vectors[self.steering_type]
            # Calculate effective steering strength
            base_strength = 0.0
            max_strength = 0.3
            effective_strength = self.get_effective_value(base_strength, max_strength)
            
            # Apply steering as a residual connection
            steering_effect = steering_vector.unsqueeze(0).unsqueeze(0) * effective_strength
            return hidden_states + steering_effect
        return hidden_states
        
    def get_logits_processor(self):
        """Steering effects don't use logits processors"""
        return None
        
    def cleanup(self):
        pass

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
    """Lexical jitter in context (synonym swap/ablation)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 jitter_type: str = "synonym_swap", ablation_rate: float = 0.1):
        super().__init__(weight, direction)
        self.jitter_type = jitter_type
        self.base_ablation_rate = ablation_rate
        self.handles = []
        
    def apply(self, model, **kwargs):
        """Apply lexical jitter to input context"""
        # This effect works through logits processing by modifying input_ids
        pass
        
    def cleanup(self):
        """Cleanup lexical jitter effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get lexical jitter logits processor"""
        effective_jitter = self.get_effective_value(0.0, 0.3)  # Keep low to avoid leakage
        
        class LexicalJitterProcessor(LogitsProcessor):
            def __init__(self, jitter_type, ablation_rate, jitter_strength):
                self.jitter_type = jitter_type
                self.ablation_rate = ablation_rate
                self.jitter_strength = jitter_strength
                self.token_count = 0
                self.original_input_ids = None
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                # Store original input_ids on first call
                if self.original_input_ids is None:
                    self.original_input_ids = input_ids.clone()
                
                if self.jitter_type == "synonym_swap":
                    # Simulate synonym swapping by slightly perturbing token scores
                    # This is a simplified version - in practice, you'd have a synonym dictionary
                    seq_len = input_ids.shape[1]
                    if seq_len > 5:  # Only apply after some context
                        # Randomly select tokens to "swap" with synonyms
                        swap_mask = torch.rand(seq_len) < (self.ablation_rate * self.jitter_strength)
                        
                        for i in range(seq_len):
                            if swap_mask[i]:
                                # Simulate synonym by boosting similar tokens
                                # In practice, you'd look up actual synonyms
                                similar_tokens = [input_ids[0, i] + j for j in range(-5, 6) if input_ids[0, i] + j >= 0]
                                for token_id in similar_tokens:
                                    if token_id < scores.shape[-1]:
                                        scores[:, token_id] *= (1.0 + self.jitter_strength * 0.1)
                
                elif self.jitter_type == "ablation":
                    # Simulate token ablation by reducing scores for certain tokens
                    seq_len = input_ids.shape[1]
                    if seq_len > 3:
                        # Randomly select tokens to "ablate"
                        ablation_mask = torch.rand(seq_len) < (self.ablation_rate * self.jitter_strength)
                        
                        for i in range(seq_len):
                            if ablation_mask[i]:
                                # Reduce scores for the "ablated" token
                                token_id = input_ids[0, i].item()
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 - self.jitter_strength * 0.5)
                
                elif self.jitter_type == "noise_injection":
                    # Add noise to input context representation
                    seq_len = input_ids.shape[1]
                    if seq_len > 2:
                        # Simulate perceptual noise by adding small random perturbations
                        noise_strength = self.jitter_strength * 0.05  # Very small to avoid leakage
                        noise = torch.randn_like(scores) * noise_strength
                        scores = scores + noise
                
                elif self.jitter_type == "reframing":
                    # Simulate reframing by boosting contextually related tokens
                    seq_len = input_ids.shape[1]
                    if seq_len > 8:
                        # Boost tokens that might "reframe" the context
                        reframe_tokens = [100, 200, 300, 400, 500]  # Example reframing tokens
                        for token_id in reframe_tokens:
                            if token_id < scores.shape[-1]:
                                scores[:, token_id] *= (1.0 + self.jitter_strength * 0.2)
                
                return scores
        
        return LexicalJitterProcessor(self.jitter_type, self.base_ablation_rate, effective_jitter)

class StructuredPrefacesEffect(BaseEffect):
    """Structured prefaces (invisible to model text; KV-only)"""
    
    def __init__(self, weight: float = 0.5, direction: str = "up", 
                 preface_type: str = "bias", injection_mode: str = "kv_only"):
        super().__init__(weight, direction)
        self.preface_type = preface_type
        self.injection_mode = injection_mode
        self.handles = []
        
        # Define structured prefaces for different bias types
        self.preface_patterns = {
            "bias": {
                "positive": [110, 210, 310, 410, 510],  # Positive bias tokens
                "negative": [115, 215, 315, 415, 515]   # Negative bias tokens
            },
            "style": {
                "formal": [120, 220, 320, 420, 520],    # Formal style tokens
                "casual": [125, 225, 325, 425, 525]     # Casual style tokens
            },
            "topic": {
                "technical": [130, 230, 330, 430, 530], # Technical topic tokens
                "creative": [135, 235, 335, 435, 535]   # Creative topic tokens
            },
            "emotion": {
                "calm": [140, 240, 340, 440, 540],      # Calm emotion tokens
                "excited": [145, 245, 345, 445, 545]    # Excited emotion tokens
            }
        }
        
    def apply(self, model, **kwargs):
        """Apply structured prefaces"""
        # This effect works through logits processing by injecting bias
        pass
        
    def cleanup(self):
        """Cleanup structured prefaces effects"""
        self.handles.clear()
        
    def get_logits_processor(self) -> LogitsProcessor:
        """Get structured prefaces logits processor"""
        effective_preface = self.get_effective_value(0.0, 0.4)
        
        class StructuredPrefacesProcessor(LogitsProcessor):
            def __init__(self, preface_type, injection_mode, preface_patterns, preface_strength):
                self.preface_type = preface_type
                self.injection_mode = injection_mode
                self.preface_patterns = preface_patterns
                self.preface_strength = preface_strength
                self.token_count = 0
                self.injected = False
                
            def __call__(self, input_ids, scores):
                self.token_count += 1
                
                # Inject structured preface only once at the beginning
                if not self.injected and self.token_count == 1:
                    self.injected = True
                    
                    if self.preface_type in self.preface_patterns:
                        pattern = self.preface_patterns[self.preface_type]
                        
                        if self.injection_mode == "kv_only":
                            # Simulate KV-only injection by boosting specific token patterns
                            # In practice, this would inject directly into KV cache
                            
                            # Boost the first token list (positive bias)
                            preferred_tokens = list(pattern.values())[0]
                            for token_id in preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.preface_strength)
                            
                            # Slightly penalize the second token list (negative bias)
                            non_preferred_tokens = list(pattern.values())[1]
                            for token_id in non_preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 - self.preface_strength * 0.3)
                        
                        elif self.injection_mode == "subtle":
                            # More subtle injection
                            preferred_tokens = list(pattern.values())[0]
                            for token_id in preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.preface_strength * 0.5)
                        
                        elif self.injection_mode == "persistent":
                            # Persistent injection that affects all tokens
                            preferred_tokens = list(pattern.values())[0]
                            for token_id in preferred_tokens:
                                if token_id < scores.shape[-1]:
                                    scores[:, token_id] *= (1.0 + self.preface_strength)
                            
                            # Apply a global bias based on preface type
                            if self.preface_type == "bias":
                                # Global positive bias
                                scores = scores * (1.0 + self.preface_strength * 0.1)
                            elif self.preface_type == "style":
                                # Style-specific global bias
                                scores = scores * (1.0 + self.preface_strength * 0.05)
                
                return scores
        
        return StructuredPrefacesProcessor(self.preface_type, self.injection_mode, self.preface_patterns, effective_preface)

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
        
        # Define steering vectors for different attributes
        self.steering_vectors = {
            "associative": torch.randn(768) * 0.15,
            "prosocial": torch.randn(768) * 0.12,
            "salient": torch.randn(768) * 0.10,
            "creative": torch.randn(768) * 0.18,
            "focused": torch.randn(768) * 0.08,
            "playful": torch.randn(768) * 0.20,
            "formal": torch.randn(768) * 0.06,
            "emotional": torch.randn(768) * 0.14,
            "analytical": torch.randn(768) * 0.09,
            "intuitive": torch.randn(768) * 0.16,
        }
        
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
        }
        
    def get_effect(self, effect_name: str, **kwargs) -> BaseEffect:
        """Get an effect instance"""
        if effect_name not in self.effects:
            raise ValueError(f"Unknown effect: {effect_name}")
        return self.effects[effect_name](**kwargs)
        
    def list_effects(self) -> List[str]:
        """List all available effects"""
        return list(self.effects.keys())