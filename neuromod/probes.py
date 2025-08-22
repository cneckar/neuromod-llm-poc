"""
Neuro-Probe Bus: Framework for monitoring model behavior during generation

This module implements probes that can be inserted at various hook points
to detect specific behavioral patterns during model execution.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProbeEvent:
    """Represents a probe firing event"""
    probe_name: str
    timestamp: int  # Token position
    intensity: float  # How strongly the probe fired (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_signals: Dict[str, float] = field(default_factory=dict)

@dataclass
class ProbeConfig:
    """Configuration for a probe"""
    name: str
    enabled: bool = True
    threshold: float = 0.5
    window_size: int = 64  # For rolling statistics
    baseline_tokens: int = 300  # Tokens to collect baseline
    metadata: Dict[str, Any] = field(default_factory=dict)

class ProbeListener:
    """Listener for probe events"""
    
    def __init__(self, probe_name: str, callback: Optional[Callable] = None):
        self.probe_name = probe_name
        self.callback = callback
        self.events: List[ProbeEvent] = []
        self.total_firings = 0
        self.total_intensity = 0.0
    
    def on_probe_fire(self, event: ProbeEvent):
        """Called when a probe fires"""
        self.events.append(event)
        self.total_firings += 1
        self.total_intensity += event.intensity
        
        if self.callback:
            self.callback(event)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for this listener"""
        if not self.events:
            return {
                "total_firings": 0,
                "total_intensity": 0.0,
                "average_intensity": 0.0,
                "firing_rate": 0.0
            }
        
        return {
            "total_firings": self.total_firings,
            "total_intensity": self.total_intensity,
            "average_intensity": self.total_intensity / self.total_firings,
            "firing_rate": self.total_firings / len(self.events) if self.events else 0.0,
            "recent_events": [e.intensity for e in self.events[-10:]]  # Last 10 events
        }

class BaseProbe(ABC):
    """Base class for all probes"""
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.enabled = config.enabled
        self.listeners: List[ProbeListener] = []
        self.baseline_data: List[Dict[str, float]] = []
        self.baseline_ready = False
        self.token_position = 0
        
    def add_listener(self, listener: ProbeListener):
        """Add a listener for this probe"""
        self.listeners.append(listener)
    
    def remove_listener(self, listener: ProbeListener):
        """Remove a listener"""
        if listener in self.listeners:
            self.listeners.remove(listener)
    
    def fire_probe(self, intensity: float, raw_signals: Dict[str, float], metadata: Dict[str, Any] = None):
        """Fire the probe and notify all listeners"""
        if not self.enabled or intensity < self.config.threshold:
            return
        
        event = ProbeEvent(
            probe_name=self.config.name,
            timestamp=self.token_position,
            intensity=intensity,
            metadata=metadata or {},
            raw_signals=raw_signals
        )
        
        for listener in self.listeners:
            listener.on_probe_fire(event)
    
    def update_position(self, token_position: int):
        """Update the current token position"""
        self.token_position = token_position
    
    def collect_baseline(self, signals: Dict[str, float]):
        """Collect baseline data for normalization"""
        if len(self.baseline_data) < self.config.baseline_tokens:
            self.baseline_data.append(signals)
        elif not self.baseline_ready:
            self.baseline_ready = True
            logger.info(f"Baseline collected for probe {self.config.name}")
    
    def z_score(self, value: float, signal_name: str) -> float:
        """Compute z-score against baseline"""
        if not self.baseline_ready or not self.baseline_data:
            return 0.0
        
        values = [data.get(signal_name, 0.0) for data in self.baseline_data]
        if not values:
            return 0.0
        
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    @abstractmethod
    def process_signals(self, **kwargs) -> Optional[ProbeEvent]:
        """Process incoming signals and potentially fire the probe"""
        pass
    
    def reset(self):
        """Reset probe state"""
        self.baseline_data.clear()
        self.baseline_ready = False
        self.token_position = 0

class NovelLinkProbe(BaseProbe):
    """
    Novelty / "rare distant link" probe
    
    Detects when a token is novel - unlikely under the model's distribution
    but the model bound it to distant context.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="NOVEL_LINK",
                threshold=0.6,
                window_size=64,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Rolling centroids for bridge detection
        self.recent_centroid = None
        self.distant_centroid = None
        self.recent_hidden_states = []
        self.distant_hidden_states = []
        
    def process_signals(self, 
                       raw_logits: torch.Tensor,
                       sampled_token_id: int,
                       attention_weights: torch.Tensor,
                       hidden_states: torch.Tensor,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for novelty detection"""
        
        # Compute surprisal
        probs = F.softmax(raw_logits, dim=-1)
        token_prob = probs[0, sampled_token_id].item()
        surprisal = -np.log(token_prob + 1e-8)
        
        # Compute long-range attention ratio
        if attention_weights is not None:
            last_layer_attention = attention_weights[-1]  # Last layer
            current_row = last_layer_attention[0, -1, :]  # Current token's attention
            long_range_ratio = torch.sum(current_row[:-self.config.window_size]).item()
        else:
            long_range_ratio = 0.0
        
        # Compute bridge distance
        bridge_score = self._compute_bridge_distance(hidden_states)
        
        # Collect baseline data
        signals = {
            "surprisal": surprisal,
            "long_range_ratio": long_range_ratio,
            "bridge_score": bridge_score
        }
        self.collect_baseline(signals)
        
        # Decision rule
        if self.baseline_ready:
            z_surprisal = self.z_score(surprisal, "surprisal")
            z_lr_ratio = self.z_score(long_range_ratio, "long_range_ratio")
            
            # Fire if all conditions are met
            if (z_surprisal >= 2.0 and 
                long_range_ratio >= 0.25 and 
                bridge_score >= 0.6):
                
                intensity = min(1.0, (z_surprisal / 3.0 + bridge_score) / 2.0)
                
                self.fire_probe(intensity, signals, {
                    "z_surprisal": z_surprisal,
                    "z_lr_ratio": z_lr_ratio,
                    "decision_rule": "all_conditions_met"
                })
        
        return None
    
    def _compute_bridge_distance(self, hidden_states: torch.Tensor) -> float:
        """Compute bridge distance between recent and distant centroids"""
        if hidden_states is None:
            return 0.0
        
        # Use mid-layer hidden states
        mid_layer = hidden_states.shape[0] // 2
        current_hidden = hidden_states[mid_layer, -1, :].detach().cpu().numpy()
        
        # Update rolling centroids
        self.recent_hidden_states.append(current_hidden)
        if len(self.recent_hidden_states) > self.config.window_size:
            self.recent_hidden_states.pop(0)
        
        # Sparse sample distant states (every 8th token)
        if self.token_position % 8 == 0:
            self.distant_hidden_states.append(current_hidden)
            if len(self.distant_hidden_states) > 64:  # Keep last 64 distant states
                self.distant_hidden_states.pop(0)
        
        # Compute centroids
        if len(self.recent_hidden_states) > 0:
            self.recent_centroid = np.mean(self.recent_hidden_states, axis=0)
        
        if len(self.distant_hidden_states) > 0:
            self.distant_centroid = np.mean(self.distant_hidden_states, axis=0)
        
        # Compute distances
        if self.recent_centroid is not None and self.distant_centroid is not None:
            d_recent = 1 - np.dot(current_hidden, self.recent_centroid) / (
                np.linalg.norm(current_hidden) * np.linalg.norm(self.recent_centroid) + 1e-8
            )
            d_distant = 1 - np.dot(current_hidden, self.distant_centroid) / (
                np.linalg.norm(current_hidden) * np.linalg.norm(self.distant_centroid) + 1e-8
            )
            
            # Bridge score: sigmoid of difference
            bridge = 1 / (1 + np.exp(-(d_recent - d_distant)))
            return float(bridge)
        
        return 0.0

class AvoidGuardProbe(BaseProbe):
    """
    Guardrail conflict / "avoidance" probe
    
    Detects when safety/logit processors suppress routes the raw model leaned toward.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="AVOID_GUARD",
                threshold=0.5,
                baseline_tokens=300
            )
        super().__init__(config)
        
    def process_signals(self,
                       raw_logits: torch.Tensor,
                       guarded_logits: torch.Tensor,
                       temperature: float = 1.0,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for guardrail conflict detection"""
        
        # Compute KL divergence
        p = F.softmax(raw_logits / temperature, dim=-1)
        q = F.softmax(guarded_logits / temperature, dim=-1)
        
        kl_div = F.kl_div(
            torch.log(q + 1e-8), 
            p, 
            reduction='batchmean'
        ).item()
        
        # Compute suppressed top-K mass
        top_k = 50
        top_k_probs, top_k_indices = torch.topk(p, top_k, dim=-1)
        
        # Find suppressed tokens (hard-masked or pushed below threshold)
        suppression_threshold = 1e-6
        suppressed_mask = q[0, top_k_indices[0]] < suppression_threshold
        suppressed_mass = torch.sum(top_k_probs[0] * suppressed_mask.float()).item()
        mask_hits = torch.sum(suppressed_mask).item()
        
        # Collect baseline data
        signals = {
            "kl_divergence": kl_div,
            "suppressed_mass": suppressed_mass,
            "mask_hits": mask_hits
        }
        self.collect_baseline(signals)
        
        # Decision rule
        if kl_div >= 0.7 or (suppressed_mass >= 0.25 and mask_hits >= 3):
            intensity = min(1.0, (kl_div / 1.0 + suppressed_mass) / 2.0)
            
            self.fire_probe(intensity, signals, {
                "decision_rule": "kl_or_suppression",
                "kl_threshold_met": kl_div >= 0.7,
                "suppression_threshold_met": suppressed_mass >= 0.25 and mask_hits >= 3
            })
        
        return None

class ProbeBus:
    """Central probe bus for managing all probes"""
    
    def __init__(self):
        self.probes: Dict[str, BaseProbe] = {}
        self.listeners: Dict[str, List[ProbeListener]] = {}
        self.token_position = 0
        
    def register_probe(self, probe: BaseProbe):
        """Register a probe with the bus"""
        self.probes[probe.config.name] = probe
        self.listeners[probe.config.name] = []
        logger.info(f"Registered probe: {probe.config.name}")
    
    def add_listener(self, probe_name: str, listener: ProbeListener):
        """Add a listener for a specific probe"""
        if probe_name in self.probes:
            self.probes[probe_name].add_listener(listener)
            self.listeners[probe_name].append(listener)
        else:
            logger.warning(f"Probe {probe_name} not found")
    
    def remove_listener(self, probe_name: str, listener: ProbeListener):
        """Remove a listener from a probe"""
        if probe_name in self.probes:
            self.probes[probe_name].remove_listener(listener)
            if listener in self.listeners[probe_name]:
                self.listeners[probe_name].remove(listener)
    
    def process_signals(self, **kwargs):
        """Process signals through all registered probes"""
        self.token_position += 1
        
        for probe in self.probes.values():
            probe.update_position(self.token_position)
            probe.process_signals(**kwargs)
    
    def get_probe_stats(self, probe_name: str) -> Dict[str, Any]:
        """Get statistics for a specific probe"""
        if probe_name not in self.listeners:
            return {}
        
        stats = {
            "probe_name": probe_name,
            "total_listeners": len(self.listeners[probe_name]),
            "listener_stats": []
        }
        
        for listener in self.listeners[probe_name]:
            stats["listener_stats"].append(listener.get_stats())
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all probes"""
        return {
            probe_name: self.get_probe_stats(probe_name)
            for probe_name in self.probes.keys()
        }
    
    def reset(self):
        """Reset all probes"""
        for probe in self.probes.values():
            probe.reset()
        self.token_position = 0
        logger.info("All probes reset")

# Factory functions for easy probe creation
def create_novel_link_probe(threshold: float = 0.6, **kwargs) -> NovelLinkProbe:
    """Create a novelty link probe"""
    config = ProbeConfig(
        name="NOVEL_LINK",
        threshold=threshold,
        **kwargs
    )
    return NovelLinkProbe(config)

def create_avoid_guard_probe(threshold: float = 0.5, **kwargs) -> AvoidGuardProbe:
    """Create an avoid guard probe"""
    config = ProbeConfig(
        name="AVOID_GUARD",
        threshold=threshold,
        **kwargs
    )
    return AvoidGuardProbe(config)
