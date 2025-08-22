"""
Updated neuromodulation tool using modular effects system
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from .pack_system import PackRegistry, PackManager, Pack
from .probes import ProbeBus, ProbeListener, create_novel_link_probe, create_avoid_guard_probe

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
            
            # Apply the pack using the pack manager
            results = self.pack_manager.apply_pack(scaled_pack, self.model)
            
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
        # Register default probes
        novel_probe = create_novel_link_probe(threshold=0.6)
        avoid_probe = create_avoid_guard_probe(threshold=0.5)
        
        self.probe_bus.register_probe(novel_probe)
        self.probe_bus.register_probe(avoid_probe)
        
        print(f"✅ Setup default probes: {list(self.probe_bus.probes.keys())}")
    
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
        self.probe_bus.process_signals(**kwargs)
    
    def reset_probes(self):
        """Reset all probes"""
        self.probe_bus.reset()

    @property
    def pulse_state(self):
        return self._pulse_state
