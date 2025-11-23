"""
Updated neuromodulation tool using modular effects system
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from .pack_system import PackRegistry, Pack, PackManager
from .probes import (
    ProbeBus, ProbeListener, 
    create_novel_link_probe, create_avoid_guard_probe,
    create_insight_consolidation_probe, create_fixation_flow_probe,
    create_working_memory_drop_probe, create_fragmentation_probe,
    create_self_inconsistency_tension_probe, create_goal_threat_probe,
    create_relief_probe, create_social_attunement_probe, create_agency_loss_probe
)
from .emotion_system import EmotionSystem

logger = logging.getLogger(__name__)

@dataclass
class NeuromodState:
    active: list
    token_pos: int

class NeuromodTool:
    def __init__(self, registry: PackRegistry, model, tokenizer, vectors=None):
        self.registry = registry
        self.model = model
        self.tokenizer = tokenizer
        self.vectors = vectors
        self.state = NeuromodState(active=[], token_pos=0)
        
        # New modular pack manager
        self.pack_manager = PackManager()
        
        # Probe system
        self.probe_bus = ProbeBus()
        self._setup_default_probes()
        
        # Emotion system
        self.emotion_system = EmotionSystem(window_size=64)
        
        # Legacy compatibility
        self.active_hooks = {}
        self._pulse_state = {"in_pulse": False}
        self.psychedelic_pack = None

    def pulse_active(self) -> bool:
        return bool(self._pulse_state.get("in_pulse", False))

    def apply(self, pack_name: str, intensity: float = 0.5, schedule=None, overrides=None):
        """Apply a pack using the new modular system"""
        try:
            # Get pack from registry
            pack = self.registry.get_pack(pack_name)
            
            # Apply intensity scaling to effect weights
            scaled_pack = self._scale_pack_intensity(pack, intensity)
            
            # Apply the pack using the pack manager (pass tokenizer for effects that need it)
            results = self.pack_manager.apply_pack(scaled_pack, self.model, tokenizer=self.tokenizer)
            
            # Store active pack
            self.state.active.append({
                "pack": scaled_pack,
                "overrides": overrides or {},
                "schedule": schedule
            })
            
            # Print results
            print(f"🎯 Applied pack '{pack_name}' with intensity {intensity}")
            for effect in results["applied_effects"]:
                print(f"   ✅ {effect['effect']} (weight={effect['weight']}, direction={effect['direction']})")
            
            if results["errors"]:
                for error in results["errors"]:
                    print(f"   ❌ {error['effect']}: {error['error']}")
            
            return {"ok": True, "active": [a["pack"].name for a in self.state.active]}
            
        except Exception as e:
            print(f"❌ Failed to apply pack {pack_name}: {e}")
            return {"ok": False, "error": str(e)}
    
    def _scale_pack_intensity(self, pack: Pack, intensity: float) -> Pack:
        """Scale all effect weights by intensity"""
        from copy import deepcopy
        from .pack_system import EffectConfig
        
        # Create a copy of the pack
        scaled_pack = deepcopy(pack)
        
        # Scale all effect weights
        for effect_config in scaled_pack.effects:
            effect_config.weight *= intensity
            effect_config.weight = max(0.0, min(1.0, effect_config.weight))
            
        return scaled_pack

    def _setup_default_probes(self):
        """Setup default probes for monitoring"""
        # Register core probes
        novel_probe = create_novel_link_probe(threshold=0.6)
        avoid_probe = create_avoid_guard_probe(threshold=0.5)
        insight_probe = create_insight_consolidation_probe(threshold=0.5)
        fixation_probe = create_fixation_flow_probe(threshold=0.4)
        memory_probe = create_working_memory_drop_probe(threshold=0.6)
        fragmentation_probe = create_fragmentation_probe(threshold=0.7)
        
        # Register new specialized probes
        self_inconsistency_probe = create_self_inconsistency_tension_probe(threshold=0.6)
        goal_threat_probe = create_goal_threat_probe(threshold=0.5)
        relief_probe = create_relief_probe(threshold=0.4)
        social_attunement_probe = create_social_attunement_probe(threshold=0.4)
        agency_loss_probe = create_agency_loss_probe(threshold=0.7)
        
        self.probe_bus.register_probe(novel_probe)
        self.probe_bus.register_probe(avoid_probe)
        self.probe_bus.register_probe(insight_probe)
        self.probe_bus.register_probe(fixation_probe)
        self.probe_bus.register_probe(memory_probe)
        self.probe_bus.register_probe(fragmentation_probe)
        self.probe_bus.register_probe(self_inconsistency_probe)
        self.probe_bus.register_probe(goal_threat_probe)
        self.probe_bus.register_probe(relief_probe)
        self.probe_bus.register_probe(social_attunement_probe)
        self.probe_bus.register_probe(agency_loss_probe)
        
        # Set up probe interactions
        self._setup_probe_interactions()
        
        print(f"✅ Setup default probes: {list(self.probe_bus.probes.keys())}")
    
    def _setup_probe_interactions(self):
        """Setup interactions between probes"""
        # Create a listener that forwards NOVEL_LINK events to INSIGHT_CONSOLIDATION
        def on_novel_link_fire(event):
            insight_probe = self.probe_bus.probes.get("INSIGHT_CONSOLIDATION")
            if insight_probe:
                insight_probe.notify_novel_link(event.timestamp, event.intensity)
        
        # Create a listener that forwards AVOID_GUARD events to GOAL_THREAT
        def on_avoid_guard_fire(event):
            goal_threat_probe = self.probe_bus.probes.get("GOAL_THREAT")
            if goal_threat_probe:
                goal_threat_probe.notify_avoid_guard(event.timestamp, event.intensity)
        
        # Create a listener that forwards INSIGHT_CONSOLIDATION events to RELIEF
        def on_insight_consolidation_fire(event):
            relief_probe = self.probe_bus.probes.get("RELIEF")
            if relief_probe:
                relief_probe.notify_insight_consolidation(event.timestamp, event.intensity)
        
        # Create a listener that forwards FRAGMENTATION events to AGENCY_LOSS
        def on_fragmentation_fire(event):
            agency_loss_probe = self.probe_bus.probes.get("AGENCY_LOSS")
            if agency_loss_probe:
                agency_loss_probe.notify_fragmentation(event.timestamp, event.intensity)
        
        # Create a listener that forwards WORKING_MEMORY_DROP events to AGENCY_LOSS
        def on_wm_drop_fire(event):
            agency_loss_probe = self.probe_bus.probes.get("AGENCY_LOSS")
            if agency_loss_probe:
                agency_loss_probe.notify_wm_drop(event.timestamp, event.intensity)
        
        # Register all listeners
        novel_listener = ProbeListener("NOVEL_LINK", on_novel_link_fire)
        avoid_guard_listener = ProbeListener("AVOID_GUARD", on_avoid_guard_fire)
        insight_listener = ProbeListener("INSIGHT_CONSOLIDATION", on_insight_consolidation_fire)
        fragmentation_listener = ProbeListener("FRAGMENTATION", on_fragmentation_fire)
        wm_drop_listener = ProbeListener("WORKING_MEMORY_DROP", on_wm_drop_fire)
        
        self.probe_bus.add_listener("NOVEL_LINK", novel_listener)
        self.probe_bus.add_listener("AVOID_GUARD", avoid_guard_listener)
        self.probe_bus.add_listener("INSIGHT_CONSOLIDATION", insight_listener)
        self.probe_bus.add_listener("FRAGMENTATION", fragmentation_listener)
        self.probe_bus.add_listener("WORKING_MEMORY_DROP", wm_drop_listener)
    
    def update_token_position(self, token_position: int):
        """Update token position for phase-based effects"""
        self.state.token_pos = token_position
        
        # Update probe positions
        self.probe_bus.token_position = token_position
        
        # Legacy psychedelic pack support
        if self.psychedelic_pack:
            self.psychedelic_pack.update_position(token_position)

    def get_logits_processors(self) -> List:
        """Get logits processors from active effects"""
        processors = []
        
        # Get processors from new modular system
        processors.extend(self.pack_manager.get_logits_processors())
        
        # Legacy support
        if self.psychedelic_pack:
            logits_processor = self.psychedelic_pack.get_logits_processor()
            if logits_processor:
                processors.append(logits_processor)
        
        return processors

    def get_generation_kwargs(self) -> Dict[str, Any]:
        """Get generation kwargs for active effects"""
        kwargs = {}
        
        # Legacy support for old system
        for hook_name, hook_data in self.active_hooks.items():
            if "_sampler" in hook_name and isinstance(hook_data, dict):
                for param, value in hook_data.items():
                    if param in ["temperature", "top_p", "presence_penalty", "frequency_penalty"]:
                        kwargs[param] = value
        
        # Legacy support for old psychedelic_temp
        if "psychedelic_temp" in self.active_hooks:
            kwargs["temperature"] = self.active_hooks["psychedelic_temp"]
        
        return kwargs
    
    def apply_steering(self, hidden_states):
        """Apply steering effects to hidden states"""
        # Apply steering from new modular system
        hidden_states = self.pack_manager.apply_steering(hidden_states)
        
        # Legacy support
        for hook_name, hook_data in self.active_hooks.items():
            if "_pack" in hook_name and hasattr(hook_data, 'apply_steering'):
                hidden_states = hook_data.apply_steering(hidden_states)
        
        return hidden_states
    
    def modify_kv_cache(self, kv_cache):
        """Modify KV cache"""
        # Apply modifications from new modular system
        kv_cache = self.pack_manager.modify_kv_cache(kv_cache)
        
        # Legacy support
        for hook_name, hook_data in self.active_hooks.items():
            if "_pack" in hook_name and hasattr(hook_data, 'modify_kv_cache'):
                kv_cache = hook_data.modify_kv_cache(kv_cache)
        
        return kv_cache

    def clear(self):
        """Clear all active effects"""
        # Clear new modular effects
        self.pack_manager.clear_effects()
        
        # Reset probes
        self.reset_probes()
        
        # Clear legacy hooks
        for h in self.active_hooks.values():
            if isinstance(h, list):
                for hh in h:
                    hh.disable()
            elif hasattr(h, 'disable'):
                h.disable()
            elif hasattr(h, 'cleanup'):
                h.cleanup()
        
        self.active_hooks = {}
        self.state.active = []
        
        # Clear psychedelic pack
        if self.psychedelic_pack:
            self.psychedelic_pack.cleanup()
            self.psychedelic_pack = None
        
        return {"ok": True}
    
    def get_effect_info(self) -> Dict[str, Any]:
        """Get information about active effects"""
        info = self.pack_manager.get_effect_info()
        
        # Add legacy info
        info["legacy_hooks"] = list(self.active_hooks.keys())
        info["active_packs"] = [a["pack"].name for a in self.state.active]
        
        return info
    
    def add_probe_listener(self, probe_name: str, callback=None) -> ProbeListener:
        """Add a listener for a specific probe"""
        listener = ProbeListener(probe_name, callback)
        self.probe_bus.add_listener(probe_name, listener)
        return listener
    
    def get_probe_stats(self, probe_name: str = None) -> Dict[str, Any]:
        """Get statistics for probes"""
        if probe_name:
            return self.probe_bus.get_probe_stats(probe_name)
        else:
            return self.probe_bus.get_all_stats()
    
    def process_probe_signals(self, **kwargs):
        """Process signals through the probe bus"""
        # Update token position
        self.probe_bus.token_position += 1
        self.token_position = self.probe_bus.token_position
        
        # Process signals through all probes
        self.probe_bus.process_signals(**kwargs)
        
        # Update emotion system with raw signals if available
        if kwargs:
            raw_signals = {}
            
            # Handle simulated probe signals (our current use case)
            if 'entropy' in kwargs:
                raw_signals['entropy'] = kwargs['entropy']
            if 'surprisal' in kwargs:
                raw_signals['surprisal'] = kwargs['surprisal']
            if 'kl_divergence' in kwargs:
                raw_signals['kl_divergence'] = kwargs['kl_divergence']
            if 'lr_attention' in kwargs:
                raw_signals['lr_attention'] = kwargs['lr_attention']
            if 'prosocial_alignment' in kwargs:
                raw_signals['prosocial_alignment'] = kwargs['prosocial_alignment']
            if 'anti_cliche_gain' in kwargs:
                raw_signals['anti_cliche_gain'] = kwargs['anti_cliche_gain']
            if 'risk_bend_mass' in kwargs:
                raw_signals['risk_bend_mass'] = kwargs['risk_bend_mass']
            
            # Handle real model signals (future use case)
            if 'raw_logits' in kwargs and kwargs['raw_logits'] is not None:
                # Compute entropy from logits
                probs = torch.softmax(kwargs['raw_logits'], dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                raw_signals['entropy'] = entropy
                
                # Compute surprisal if we have sampled token
                if 'sampled_token_id' in kwargs:
                    token_prob = probs[0, kwargs['sampled_token_id']].item()
                    surprisal = -np.log(token_prob + 1e-8)
                    raw_signals['surprisal'] = surprisal
            
            # Update emotion system with any available signals
            if raw_signals:
                print(f"🔍 Updating emotion system with signals: {list(raw_signals.keys())}")
                self.emotion_system.update_raw_signals(raw_signals)
            else:
                print(f"⚠️  No valid signals found in kwargs: {list(kwargs.keys())}")
    
    def reset_probes(self):
        """Reset all probes"""
        self.probe_bus.reset()
        self.emotion_system.reset()
    
    def get_emotion_state(self) -> Optional[Dict[str, Any]]:
        """Get current emotional state"""
        state = self.emotion_system.get_current_emotion_state()
        if state is None:
            return None
        
        return {
            'timestamp': state.timestamp,
            'token_position': state.token_position,
            'latent_axes': {
                'arousal': state.arousal,
                'valence': state.valence,
                'certainty': state.certainty,
                'openness': state.openness,
                'integration': state.integration,
                'sociality': state.sociality,
                'risk_preference': state.risk_preference
            },
            'emotions': state.emotions,
            'dominant_emotions': self.emotion_system.get_dominant_emotions(top_k=3),
            'probe_stats': state.probe_stats
        }
    
    def get_emotion_history(self, window_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get emotion history"""
        history = self.emotion_system.get_emotion_history(window_size)
        return [
            {
                'timestamp': state.timestamp,
                'token_position': state.token_position,
                'latent_axes': {
                    'arousal': state.arousal,
                    'valence': state.valence,
                    'certainty': state.certainty,
                    'openness': state.openness,
                    'integration': state.integration,
                    'sociality': state.sociality,
                    'risk_preference': state.risk_preference
                },
                'emotions': state.emotions,
                'dominant_emotions': [
                    (name, prob, intensity) 
                    for name, prob, intensity in self.emotion_system.get_dominant_emotions(top_k=3)
                ]
            }
            for state in history
        ]
    
    def update_emotion_state(self) -> Optional[Dict[str, Any]]:
        """Manually update emotion state and return it"""
        state = self.emotion_system.update_emotion_state(self.token_position)
        return self.get_emotion_state()

    @property
    def pulse_state(self):
        return self._pulse_state

    def register_probe_hooks(self, model):
        """Register probe hooks on the model for real-time signal capture"""
        if not hasattr(self, 'probe_hooks_registered') or not self.probe_hooks_registered:
            self.probe_hooks_registered = True
            self.model_hooks = []
            
            # Register hooks on attention layers to capture probe signals
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    # Hook into attention output
                    hook = module.register_forward_hook(
                        lambda mod, inp, output, name=name: self._attention_hook(mod, inp, output, name)
                    )
                    self.model_hooks.append(hook)
                    
                elif 'mlp' in name.lower() or 'feed_forward' in name.lower():
                    # Hook into MLP output
                    hook = module.register_forward_hook(
                        lambda mod, inp, output, name=name: self._mlp_hook(mod, inp, output, name)
                    )
                    self.model_hooks.append(hook)
                    
                elif 'layer_norm' in name.lower():
                    # Hook into layer norm output
                    hook = module.register_forward_hook(
                        lambda mod, inp, output, name=name: self._layer_norm_hook(mod, inp, output, name)
                    )
                    self.model_hooks.append(hook)
            
            logger.info(f"Registered {len(self.model_hooks)} probe hooks on model")
    
    def _attention_hook(self, module, input, output, name):
        """Hook for attention layers to capture probe signals"""
        if hasattr(output, 'last_hidden_state'):
            hidden_states = output.last_hidden_state
        elif isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        if hidden_states is not None and hidden_states.numel() > 0:
            # Extract attention patterns and compute probe signals
            signals = self._extract_attention_signals(hidden_states, name)
            self.process_probe_signals(**signals)
    
    def _mlp_hook(self, module, input, output, name):
        """Hook for MLP layers to capture probe signals"""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        if hidden_states is not None and hidden_states.numel() > 0:
            # Extract MLP patterns and compute probe signals
            signals = self._extract_mlp_signals(hidden_states, name)
            self.process_probe_signals(**signals)
    
    def _layer_norm_hook(self, module, input, output, name):
        """Hook for layer norm layers to capture probe signals"""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        if hidden_states is not None and hidden_states.numel() > 0:
            # Extract normalization patterns and compute probe signals
            signals = self._extract_norm_signals(hidden_states, name)
            self.process_probe_signals(**signals)
    
    def _extract_attention_signals(self, hidden_states, layer_name):
        """Extract probe signals from attention hidden states"""
        signals = {}
        
        # Compute entropy from hidden states
        if hidden_states.dim() > 1:
            # Compute variance across sequence dimension
            seq_variance = torch.var(hidden_states, dim=1).mean().item()
            signals['attention_variance'] = seq_variance
            
            # Compute attention focus (how concentrated the attention is)
            attention_focus = torch.softmax(hidden_states.mean(dim=-1), dim=-1).max().item()
            signals['attention_focus'] = attention_focus
            
            # Compute novelty (how different from previous states)
            if hasattr(self, '_prev_attention_states'):
                novelty = torch.norm(hidden_states - self._prev_attention_states).item()
                signals['attention_novelty'] = novelty
            self._prev_attention_states = hidden_states.detach().clone()
        
        return signals
    
    def _extract_mlp_signals(self, hidden_states, layer_name):
        """Extract probe signals from MLP hidden states"""
        signals = {}
        
        if hidden_states.dim() > 1:
            # Compute activation sparsity
            sparsity = (hidden_states == 0).float().mean().item()
            signals['mlp_sparsity'] = sparsity
            
            # Compute activation magnitude
            magnitude = torch.norm(hidden_states).item()
            signals['mlp_magnitude'] = magnitude
            
            # Compute feature diversity
            feature_diversity = torch.std(hidden_states.mean(dim=1)).item()
            signals['mlp_diversity'] = feature_diversity
        
        return signals
    
    def _extract_norm_signals(self, hidden_states, layer_name):
        """Extract probe signals from normalization layers"""
        signals = {}
        
        if hidden_states.dim() > 1:
            # Compute normalization statistics
            mean_val = hidden_states.mean().item()
            std_val = hidden_states.std().item()
            signals['norm_mean'] = mean_val
            signals['norm_std'] = std_val
            
            # Compute normalization stability
            if hasattr(self, '_prev_norm_stats'):
                prev_mean, prev_std = self._prev_norm_stats
                stability = abs(mean_val - prev_mean) + abs(std_val - prev_std)
                signals['norm_stability'] = stability
            self._prev_norm_stats = (mean_val, std_val)
        
        return signals
    
    def remove_probe_hooks(self):
        """Remove all registered probe hooks"""
        if hasattr(self, 'model_hooks'):
            for hook in self.model_hooks:
                hook.remove()
            self.model_hooks = []
            self.probe_hooks_registered = False
            logger.info("Removed all probe hooks")
