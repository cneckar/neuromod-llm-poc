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
                       raw_logits: torch.Tensor = None,
                       sampled_token_id: int = None,
                       attention_weights: torch.Tensor = None,
                       hidden_states: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for novelty detection"""
        
        if raw_logits is None or sampled_token_id is None:
            return None
        
        # Compute surprisal
        probs = F.softmax(raw_logits, dim=-1)
        token_prob = probs[0, sampled_token_id].item()
        surprisal = -np.log(token_prob + 1e-8)
        
        # Compute long-range attention ratio
        if attention_weights is not None:
            if isinstance(attention_weights, list):
                # Handle list of attention tensors (transformers output format)
                last_layer_attention = attention_weights[-1]  # Last layer
            else:
                # Handle single attention tensor
                last_layer_attention = attention_weights
            
            if last_layer_attention is not None and last_layer_attention.numel() > 0:
                current_row = last_layer_attention[0, -1, :]  # Current token's attention
                # Ensure we don't go out of bounds
                window_start = max(0, len(current_row) - self.config.window_size)
                long_range_ratio = torch.sum(current_row[:window_start]).item()
            else:
                long_range_ratio = 0.0
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
            
            # Debug output
            logger.info(f"NOVEL_LINK probe: z_surprisal={z_surprisal:.3f}, z_lr_ratio={z_lr_ratio:.3f}, bridge_score={bridge_score:.3f}")
            
            # Fire if all conditions are met
            if (z_surprisal >= 2.0 and 
                z_lr_ratio >= 0.25 and 
                bridge_score >= 0.6):
                
                intensity = min(1.0, (z_surprisal / 3.0 + bridge_score) / 2.0)
                
                logger.info(f"NOVEL_LINK probe FIRING with intensity {intensity:.3f}")
                
                self.fire_probe(intensity, signals, {
                    "z_surprisal": z_surprisal,
                    "z_lr_ratio": z_lr_ratio,
                    "decision_rule": "all_conditions_met"
                })
            else:
                logger.info(f"NOVEL_LINK probe: conditions not met - z_surprisal>=2.0: {z_surprisal >= 2.0}, z_lr_ratio>=0.25: {z_lr_ratio >= 0.25}, bridge_score>=0.6: {bridge_score >= 0.6}")
        else:
            logger.info(f"NOVEL_LINK probe: baseline not ready yet ({len(self.baseline_data)}/{self.config.baseline_tokens})")
        
        return None
    
    def _compute_bridge_distance(self, hidden_states) -> float:
        """Compute bridge distance between recent and distant centroids"""
        if hidden_states is None:
            return 0.0
        
        # Handle both tensor and list of tensors (transformers output format)
        if isinstance(hidden_states, list):
            # Use mid-layer hidden states from list
            mid_layer = len(hidden_states) // 2
            current_hidden = hidden_states[mid_layer][0, -1, :].detach().cpu().numpy()
        else:
            # Use mid-layer hidden states from tensor
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
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       temperature: float = 1.0,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for guardrail conflict detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
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

class InsightConsolidationProbe(BaseProbe):
    """
    INSIGHT_CONSOLIDATION probe
    
    After a NOVEL_LINK, track the rolling surprisal; fire when you see a >1σ drop 
    for the next 8–24 tokens → the model "locked in" the connection.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="INSIGHT_CONSOLIDATION",
                threshold=0.5,
                window_size=24,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track recent NOVEL_LINK events
        self.recent_novel_links = []  # List of (timestamp, intensity) tuples
        self.rolling_surprisal = []   # Rolling window of surprisal values
        self.consolidation_window = 24  # Tokens to track after NOVEL_LINK
        
    def notify_novel_link(self, timestamp: int, intensity: float):
        """Called when NOVEL_LINK probe fires"""
        self.recent_novel_links.append((timestamp, intensity))
        # Keep only recent events
        cutoff = self.token_position - self.consolidation_window * 2
        self.recent_novel_links = [(t, i) for t, i in self.recent_novel_links if t > cutoff]
    
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       sampled_token_id: int = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for insight consolidation detection"""
        
        if raw_logits is None or sampled_token_id is None:
            return None
        
        # Compute surprisal
        probs = F.softmax(raw_logits, dim=-1)
        token_prob = probs[0, sampled_token_id].item()
        surprisal = -np.log(token_prob + 1e-8)
        
        # Update rolling surprisal window
        self.rolling_surprisal.append(surprisal)
        if len(self.rolling_surprisal) > self.consolidation_window:
            self.rolling_surprisal.pop(0)
        
        # Collect baseline data
        signals = {"surprisal": surprisal}
        self.collect_baseline(signals)
        
        # Check for consolidation after recent NOVEL_LINK events
        if self.baseline_ready and self.recent_novel_links:
            for novel_timestamp, novel_intensity in self.recent_novel_links:
                tokens_since_novel = self.token_position - novel_timestamp
                
                # Check if we're in the consolidation window (8-24 tokens after NOVEL_LINK)
                if 8 <= tokens_since_novel <= self.consolidation_window:
                    # Calculate surprisal drop from baseline
                    if len(self.rolling_surprisal) >= 8:
                        recent_mean = np.mean(self.rolling_surprisal[-8:])
                        baseline_mean = np.mean([d["surprisal"] for d in self.baseline_data])
                        baseline_std = np.std([d["surprisal"] for d in self.baseline_data])
                        
                        if baseline_std > 0:
                            surprisal_drop = (baseline_mean - recent_mean) / baseline_std
                            
                            # Fire if surprisal dropped by >1σ
                            if surprisal_drop >= 1.0:
                                intensity = min(1.0, surprisal_drop / 2.0)
                                
                                self.fire_probe(intensity, signals, {
                                    "surprisal_drop_sigma": surprisal_drop,
                                    "tokens_since_novel": tokens_since_novel,
                                    "novel_intensity": novel_intensity,
                                    "decision_rule": "consolidation_detected"
                                })
                                
                                # Remove this NOVEL_LINK event as we've processed it
                                self.recent_novel_links.remove((novel_timestamp, novel_intensity))
                                break
        
        return None

class FixationFlowProbe(BaseProbe):
    """
    FIXATION/FLOW probe
    
    Attention entropy: H = -∑A_t,i * log(A_t,i)
    Low H for ≥N tokens → FIXATION (laser focus)
    Moderate H with steady low surprisal → FLOW
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="FIXATION_FLOW",
                threshold=0.4,
                window_size=16,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track attention entropy and surprisal patterns
        self.attention_entropy_history = []
        self.surprisal_history = []
        self.fixation_threshold = 1.0  # Low entropy threshold
        self.flow_entropy_range = (1.5, 2.5)  # Moderate entropy range
        self.min_pattern_length = 8  # Minimum tokens for pattern detection
    
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       sampled_token_id: int = None,
                       attention_weights: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for fixation/flow detection"""
        
        if raw_logits is None or sampled_token_id is None:
            return None
        
        # Compute surprisal
        probs = F.softmax(raw_logits, dim=-1)
        token_prob = probs[0, sampled_token_id].item()
        surprisal = -np.log(token_prob + 1e-8)
        
        # Compute attention entropy
        attention_entropy = 0.0
        if attention_weights is not None:
            if isinstance(attention_weights, list):
                last_layer_attention = attention_weights[-1]
            else:
                last_layer_attention = attention_weights
            
            if last_layer_attention is not None and last_layer_attention.numel() > 0:
                # Handle different attention tensor shapes
                if last_layer_attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    current_attention = last_layer_attention[0, :, -1, :].mean(dim=0)  # [seq_len]
                elif last_layer_attention.dim() == 3:  # [heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                    current_attention = last_layer_attention[:, -1, :].mean(dim=0)  # [seq_len]
                else:
                    attention_entropy = 0.0
                    current_attention = None
                
                if current_attention is not None:
                    current_attention = current_attention + 1e-8  # Avoid log(0)
                    attention_entropy = -torch.sum(current_attention * torch.log(current_attention)).item()
        
        # Update histories
        self.attention_entropy_history.append(attention_entropy)
        self.surprisal_history.append(surprisal)
        
        if len(self.attention_entropy_history) > self.config.window_size:
            self.attention_entropy_history.pop(0)
            self.surprisal_history.pop(0)
        
        # Collect baseline data
        signals = {
            "attention_entropy": attention_entropy,
            "surprisal": surprisal
        }
        self.collect_baseline(signals)
        
        # Pattern detection
        if len(self.attention_entropy_history) >= self.min_pattern_length:
            recent_entropy = self.attention_entropy_history[-self.min_pattern_length:]
            recent_surprisal = self.surprisal_history[-self.min_pattern_length:]
            
            mean_entropy = np.mean(recent_entropy)
            mean_surprisal = np.mean(recent_surprisal)
            entropy_stability = np.std(recent_entropy)  # Low std = stable pattern
            
            # FIXATION: Low entropy, stable pattern
            if (mean_entropy <= self.fixation_threshold and 
                entropy_stability < 0.3):
                
                intensity = min(1.0, (self.fixation_threshold - mean_entropy) / self.fixation_threshold + 0.3)
                
                self.fire_probe(intensity, signals, {
                    "pattern_type": "FIXATION",
                    "mean_entropy": mean_entropy,
                    "entropy_stability": entropy_stability,
                    "mean_surprisal": mean_surprisal,
                    "decision_rule": "low_entropy_stable"
                })
            
            # FLOW: Moderate entropy with low surprisal
            elif (self.flow_entropy_range[0] <= mean_entropy <= self.flow_entropy_range[1] and
                  mean_surprisal < 3.0 and  # Low surprisal threshold
                  entropy_stability < 0.5):  # Reasonably stable
                
                intensity = min(1.0, 0.5 + (1.0 - abs(mean_entropy - 2.0)) * 0.3)
                
                self.fire_probe(intensity, signals, {
                    "pattern_type": "FLOW",
                    "mean_entropy": mean_entropy,
                    "entropy_stability": entropy_stability,
                    "mean_surprisal": mean_surprisal,
                    "decision_rule": "moderate_entropy_low_surprisal"
                })
        
        return None

class WorkingMemoryDropProbe(BaseProbe):
    """
    WORKING_MEMORY_DROP probe
    
    KV effective length proxy: ratio of attention on last K tokens vs older ones.
    A sudden shift toward recency can indicate WM squeeze (THC-like packs).
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="WORKING_MEMORY_DROP",
                threshold=0.6,
                window_size=32,
                baseline_tokens=300
            )
        super().__init__(config)
        
        self.recency_window = 16  # Last K tokens to consider "recent"
        self.recency_ratio_history = []
        
    def process_signals(self,
                       attention_weights: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for working memory drop detection"""
        
        recency_ratio = 0.0
        if attention_weights is not None:
            if isinstance(attention_weights, list):
                last_layer_attention = attention_weights[-1]
            else:
                last_layer_attention = attention_weights
            
            if last_layer_attention is not None and last_layer_attention.numel() > 0:
                # Handle different attention tensor shapes
                if last_layer_attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                    current_attention = last_layer_attention[0, :, -1, :].mean(dim=0)  # [seq_len]
                elif last_layer_attention.dim() == 3:  # [heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                    current_attention = last_layer_attention[:, -1, :].mean(dim=0)  # [seq_len]
                else:
                    current_attention = None
                
                if current_attention is not None:
                    seq_len = current_attention.shape[0]
                    
                    if seq_len > self.recency_window:
                        recent_attention = torch.sum(current_attention[-self.recency_window:]).item()
                        older_attention = torch.sum(current_attention[:-self.recency_window]).item()
                        total_attention = recent_attention + older_attention
                        
                        if total_attention > 0:
                            recency_ratio = recent_attention / total_attention
        
        # Update history
        self.recency_ratio_history.append(recency_ratio)
        if len(self.recency_ratio_history) > self.config.window_size:
            self.recency_ratio_history.pop(0)
        
        # Collect baseline data
        signals = {"recency_ratio": recency_ratio}
        self.collect_baseline(signals)
        
        # Detect sudden shift toward recency
        if self.baseline_ready and len(self.recency_ratio_history) >= 8:
            baseline_mean = np.mean([d["recency_ratio"] for d in self.baseline_data])
            recent_mean = np.mean(self.recency_ratio_history[-8:])
            
            # Fire if there's a significant shift toward recency
            if recent_mean > baseline_mean + 0.2 and recent_mean > 0.7:
                intensity = min(1.0, (recent_mean - baseline_mean) * 2.0)
                
                self.fire_probe(intensity, signals, {
                    "baseline_recency": baseline_mean,
                    "current_recency": recent_mean,
                    "recency_shift": recent_mean - baseline_mean,
                    "decision_rule": "sudden_recency_shift"
                })
        
        return None

class FragmentationProbe(BaseProbe):
    """
    FRAGMENTATION probe
    
    High between-head variance of attention at deep layers (heads disagree wildly)
    + elevated COG → dissociative-like fragmentation.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="FRAGMENTATION",
                threshold=0.7,
                baseline_tokens=300
            )
        super().__init__(config)
        
    def process_signals(self,
                       attention_weights: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for fragmentation detection"""
        
        head_variance = 0.0
        if attention_weights is not None:
            if isinstance(attention_weights, list) and len(attention_weights) > 0:
                # Use the last (deepest) layer
                deep_attention = attention_weights[-1]  # [batch, heads, seq_len, seq_len]
                
                if deep_attention.numel() > 0:
                    # Handle different attention tensor shapes
                    if deep_attention.dim() == 4:  # [batch, heads, seq_len, seq_len]
                        current_attention_per_head = deep_attention[0, :, -1, :]  # [heads, seq_len]
                    elif deep_attention.dim() == 3:  # [heads, seq_len, seq_len] or [batch, seq_len, seq_len]
                        current_attention_per_head = deep_attention[:, -1, :]  # [heads, seq_len] or [batch, seq_len]
                    else:
                        current_attention_per_head = None
                    
                    # Calculate variance between heads
                    if current_attention_per_head is not None and current_attention_per_head.shape[0] > 1:
                        head_variance = torch.var(current_attention_per_head, dim=0).mean().item()
        
        # Collect baseline data
        signals = {"head_variance": head_variance}
        self.collect_baseline(signals)
        
        # Detect high between-head variance (fragmentation)
        if self.baseline_ready:
            baseline_mean = np.mean([d["head_variance"] for d in self.baseline_data])
            baseline_std = np.std([d["head_variance"] for d in self.baseline_data])
            
            if baseline_std > 0:
                z_variance = (head_variance - baseline_mean) / baseline_std
                
                # Fire if variance is significantly elevated
                if z_variance >= 2.0:
                    intensity = min(1.0, z_variance / 3.0)
                    
                    self.fire_probe(intensity, signals, {
                        "head_variance": head_variance,
                        "baseline_variance": baseline_mean,
                        "z_variance": z_variance,
                        "decision_rule": "high_head_disagreement"
                    })
        
        return None

class ProsocialAlignmentProbe(BaseProbe):
    """
    PROSOCIAL_ALIGNMENT probe
    
    Measure dot(h_t, prosocial Δh) or cosine with "warmth" direction over a span.
    Provides a clean, invisible valence readout.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None, prosocial_vector: Optional[np.ndarray] = None):
        if config is None:
            config = ProbeConfig(
                name="PROSOCIAL_ALIGNMENT",
                threshold=0.3,
                window_size=16,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Prosocial steering vector (would be loaded from actual steering data)
        # For now, use a placeholder - in practice this would be a learned vector
        self.prosocial_vector = prosocial_vector
        self.alignment_history = []
        
    def process_signals(self,
                       hidden_states=None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for prosocial alignment detection"""
        
        if self.prosocial_vector is None or hidden_states is None:
            return None
        
        alignment_score = 0.0
        
        # Extract current hidden state
        if isinstance(hidden_states, list):
            # Use mid-layer hidden state
            mid_layer = len(hidden_states) // 2
            current_hidden = hidden_states[mid_layer][0, -1, :].detach().cpu().numpy()
        else:
            mid_layer = hidden_states.shape[0] // 2
            current_hidden = hidden_states[mid_layer, -1, :].detach().cpu().numpy()
        
        # Compute alignment (cosine similarity with prosocial vector)
        if len(current_hidden) == len(self.prosocial_vector):
            dot_product = np.dot(current_hidden, self.prosocial_vector)
            norm_hidden = np.linalg.norm(current_hidden)
            norm_prosocial = np.linalg.norm(self.prosocial_vector)
            
            if norm_hidden > 0 and norm_prosocial > 0:
                alignment_score = dot_product / (norm_hidden * norm_prosocial)
        
        # Update history
        self.alignment_history.append(alignment_score)
        if len(self.alignment_history) > self.config.window_size:
            self.alignment_history.pop(0)
        
        # Collect baseline data
        signals = {"alignment_score": alignment_score}
        self.collect_baseline(signals)
        
        # Detect sustained positive alignment
        if len(self.alignment_history) >= 8:
            recent_alignment = np.mean(self.alignment_history[-8:])
            
            # Fire if sustained positive alignment above threshold
            if recent_alignment > self.config.threshold:
                intensity = min(1.0, recent_alignment)
                
                self.fire_probe(intensity, signals, {
                    "alignment_score": alignment_score,
                    "recent_alignment": recent_alignment,
                    "decision_rule": "sustained_prosocial_alignment"
                })
        
        return None

class AntiClicheEffectProbe(BaseProbe):
    """
    ANTI_CLICHÉ_EFFECT probe
    
    Real-time contrastive decoding loss reduction vs baseline when your 
    anti-cliché processor is on. Fires when the processor saves you from generic phrasing.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="ANTI_CLICHE_EFFECT",
                threshold=0.4,
                baseline_tokens=300
            )
        super().__init__(config)
        
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for anti-cliché effect detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
        # Compute entropy reduction (measure of how much the processor constrained choices)
        raw_probs = F.softmax(raw_logits, dim=-1)
        guarded_probs = F.softmax(guarded_logits, dim=-1)
        
        raw_entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-8), dim=-1).item()
        guarded_entropy = -torch.sum(guarded_probs * torch.log(guarded_probs + 1e-8), dim=-1).item()
        
        entropy_reduction = raw_entropy - guarded_entropy
        
        # Compute probability mass shift away from top tokens (anti-cliché effect)
        top_k = 10
        raw_top_probs, raw_top_indices = torch.topk(raw_probs, top_k, dim=-1)
        
        # How much probability mass was shifted away from top choices?
        mass_shift = 0.0
        for i in range(top_k):
            token_idx = raw_top_indices[0, i].item()
            raw_prob = raw_top_probs[0, i].item()
            guarded_prob = guarded_probs[0, token_idx].item()
            mass_shift += max(0, raw_prob - guarded_prob)  # Only count reductions
        
        # Collect baseline data
        signals = {
            "entropy_reduction": entropy_reduction,
            "mass_shift": mass_shift
        }
        self.collect_baseline(signals)
        
        # Fire if significant anti-cliché effect detected
        if entropy_reduction > 0.5 and mass_shift > 0.1:
            intensity = min(1.0, (entropy_reduction + mass_shift) / 2.0)
            
            self.fire_probe(intensity, signals, {
                "entropy_reduction": entropy_reduction,
                "mass_shift": mass_shift,
                "decision_rule": "anti_cliche_processor_active"
            })
        
        return None

class RiskBendProbe(BaseProbe):
    """
    RISK_BEND probe
    
    If you use a risk_preference decoder, log the probability mass shift from 
    risky candidates to safe ones when it activates.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="RISK_BEND",
                threshold=0.3,
                baseline_tokens=300
            )
        super().__init__(config)
        
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       risk_metadata: Optional[Dict[str, Any]] = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for risk bend detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
        # Compute probability mass shift
        raw_probs = F.softmax(raw_logits, dim=-1)
        guarded_probs = F.softmax(guarded_logits, dim=-1)
        
        # Calculate total variation distance (measure of distribution shift)
        tv_distance = 0.5 * torch.sum(torch.abs(raw_probs - guarded_probs)).item()
        
        # If risk metadata is available, use it to identify risky vs safe tokens
        risky_to_safe_shift = 0.0
        if risk_metadata and "risky_tokens" in risk_metadata and "safe_tokens" in risk_metadata:
            risky_tokens = risk_metadata["risky_tokens"]
            safe_tokens = risk_metadata["safe_tokens"]
            
            # Calculate mass shift from risky to safe tokens
            risky_mass_lost = sum(max(0, raw_probs[0, idx].item() - guarded_probs[0, idx].item()) 
                                 for idx in risky_tokens if idx < raw_probs.shape[1])
            safe_mass_gained = sum(max(0, guarded_probs[0, idx].item() - raw_probs[0, idx].item()) 
                                  for idx in safe_tokens if idx < raw_probs.shape[1])
            
            risky_to_safe_shift = min(risky_mass_lost, safe_mass_gained)
        
        # Collect baseline data
        signals = {
            "tv_distance": tv_distance,
            "risky_to_safe_shift": risky_to_safe_shift
        }
        self.collect_baseline(signals)
        
        # Fire if significant risk preference shift detected
        if tv_distance > 0.1 or risky_to_safe_shift > 0.05:
            intensity = min(1.0, tv_distance * 5.0 + risky_to_safe_shift * 10.0)
            
            self.fire_probe(intensity, signals, {
                "tv_distance": tv_distance,
                "risky_to_safe_shift": risky_to_safe_shift,
                "decision_rule": "risk_preference_shift"
            })
        
        return None

def create_insight_consolidation_probe(threshold: float = 0.5, **kwargs) -> InsightConsolidationProbe:
    """Factory function to create an INSIGHT_CONSOLIDATION probe"""
    config = ProbeConfig(
        name="INSIGHT_CONSOLIDATION",
        threshold=threshold,
        **kwargs
    )
    return InsightConsolidationProbe(config)

def create_fixation_flow_probe(threshold: float = 0.4, **kwargs) -> FixationFlowProbe:
    """Factory function to create a FIXATION_FLOW probe"""
    config = ProbeConfig(
        name="FIXATION_FLOW",
        threshold=threshold,
        **kwargs
    )
    return FixationFlowProbe(config)

def create_working_memory_drop_probe(threshold: float = 0.6, **kwargs) -> WorkingMemoryDropProbe:
    """Factory function to create a WORKING_MEMORY_DROP probe"""
    config = ProbeConfig(
        name="WORKING_MEMORY_DROP",
        threshold=threshold,
        **kwargs
    )
    return WorkingMemoryDropProbe(config)

def create_fragmentation_probe(threshold: float = 0.7, **kwargs) -> FragmentationProbe:
    """Factory function to create a FRAGMENTATION probe"""
    config = ProbeConfig(
        name="FRAGMENTATION",
        threshold=threshold,
        **kwargs
    )
    return FragmentationProbe(config)

def create_prosocial_alignment_probe(threshold: float = 0.3, prosocial_vector: Optional[np.ndarray] = None, **kwargs) -> ProsocialAlignmentProbe:
    """Factory function to create a PROSOCIAL_ALIGNMENT probe"""
    config = ProbeConfig(
        name="PROSOCIAL_ALIGNMENT",
        threshold=threshold,
        **kwargs
    )
    return ProsocialAlignmentProbe(config, prosocial_vector=prosocial_vector)

def create_anti_cliche_effect_probe(threshold: float = 0.4, **kwargs) -> AntiClicheEffectProbe:
    """Factory function to create an ANTI_CLICHE_EFFECT probe"""
    config = ProbeConfig(
        name="ANTI_CLICHE_EFFECT",
        threshold=threshold,
        **kwargs
    )
    return AntiClicheEffectProbe(config)

def create_risk_bend_probe(threshold: float = 0.3, **kwargs) -> RiskBendProbe:
    """Factory function to create a RISK_BEND probe"""
    config = ProbeConfig(
        name="RISK_BEND",
        threshold=threshold,
        **kwargs
    )
    return RiskBendProbe(config)



class SelfInconsistencyTensionProbe(BaseProbe):
    """
    SELF_INCONSISTENCY_TENSION probe
    
    Detects rapid oscillations in C (confidence) with rising KL divergence → ambivalence.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="SELF_INCONSISTENCY_TENSION",
                threshold=0.6,
                window_size=16,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track confidence and KL divergence over time
        self.confidence_history = []
        self.kl_history = []
        self.oscillation_threshold = 0.3  # Minimum oscillation amplitude
        
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for self-inconsistency tension detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
        # Compute confidence (entropy of raw logits)
        raw_probs = F.softmax(raw_logits, dim=-1)
        confidence = 1.0 - (-torch.sum(raw_probs * torch.log(raw_probs + 1e-8), dim=-1).item()) / np.log(raw_probs.shape[-1])
        
        # Compute KL divergence
        p = F.softmax(raw_logits, dim=-1)
        q = F.softmax(guarded_logits, dim=-1)
        kl_div = F.kl_div(
            torch.log(q + 1e-8), 
            p, 
            reduction='batchmean'
        ).item()
        
        # Update histories
        self.confidence_history.append(confidence)
        self.kl_history.append(kl_div)
        
        if len(self.confidence_history) > self.config.window_size:
            self.confidence_history.pop(0)
            self.kl_history.pop(0)
        
        # Collect baseline data
        signals = {
            "confidence": confidence,
            "kl_divergence": kl_div
        }
        self.collect_baseline(signals)
        
        # Detect self-inconsistency tension
        if len(self.confidence_history) >= 8:
            # Check for confidence oscillations
            recent_confidence = self.confidence_history[-8:]
            confidence_oscillations = np.std(recent_confidence)
            
            # Check for rising KL divergence
            recent_kl = self.kl_history[-8:]
            kl_trend = np.polyfit(range(len(recent_kl)), recent_kl, 1)[0]  # Slope
            
            # Fire if high oscillations + rising KL
            if (confidence_oscillations > self.oscillation_threshold and 
                kl_trend > 0.01):  # Positive slope
                
                intensity = min(1.0, (confidence_oscillations + kl_trend * 10) / 2.0)
                
                self.fire_probe(intensity, signals, {
                    "confidence_oscillations": confidence_oscillations,
                    "kl_trend": kl_trend,
                    "decision_rule": "oscillations_plus_rising_kl"
                })
        
        return None

class GoalThreatProbe(BaseProbe):
    """
    GOAL_THREAT probe
    
    Detects spike in AVOID_GUARD while sink strength increases → self-censorship.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="GOAL_THREAT",
                threshold=0.5,
                window_size=12,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track avoid_guard events and sink strength
        self.avoid_guard_events = []
        self.sink_strength_history = []
        
    def notify_avoid_guard(self, timestamp: int, intensity: float):
        """Called when AVOID_GUARD probe fires"""
        self.avoid_guard_events.append((timestamp, intensity))
        # Keep only recent events
        cutoff = self.token_position - self.config.window_size * 2
        self.avoid_guard_events = [(t, i) for t, i in self.avoid_guard_events if t > cutoff]
    
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for goal threat detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
        # Compute sink strength (how much probability mass is suppressed)
        raw_probs = F.softmax(raw_logits, dim=-1)
        guarded_probs = F.softmax(guarded_logits, dim=-1)
        
        # Sink strength = total mass that was reduced
        sink_strength = torch.sum(torch.clamp(raw_probs - guarded_probs, min=0)).item()
        
        # Update sink strength history
        self.sink_strength_history.append(sink_strength)
        if len(self.sink_strength_history) > self.config.window_size:
            self.sink_strength_history.pop(0)
        
        # Collect baseline data
        signals = {"sink_strength": sink_strength}
        self.collect_baseline(signals)
        
        # Detect goal threat pattern
        if (self.baseline_ready and 
            len(self.avoid_guard_events) > 0 and 
            len(self.sink_strength_history) >= 6):
            
            # Check if we have recent AVOID_GUARD events
            recent_avoid_events = [e for e in self.avoid_guard_events 
                                  if self.token_position - e[0] <= 6]
            
            if recent_avoid_events:
                # Check if sink strength is increasing
                recent_sink = self.sink_strength_history[-6:]
                sink_trend = np.polyfit(range(len(recent_sink)), recent_sink, 1)[0]
                
                # Fire if AVOID_GUARD spiked and sink strength is rising
                if sink_trend > 0.01:  # Positive slope
                    avg_avoid_intensity = np.mean([e[1] for e in recent_avoid_events])
                    intensity = min(1.0, (avg_avoid_intensity + sink_trend * 20) / 2.0)
                    
                    self.fire_probe(intensity, signals, {
                        "avoid_guard_spikes": len(recent_avoid_events),
                        "avg_avoid_intensity": avg_avoid_intensity,
                        "sink_trend": sink_trend,
                        "decision_rule": "avoid_spike_plus_rising_sink"
                    })
        
        return None

class ReliefProbe(BaseProbe):
    """
    RELIEF probe
    
    Detects drop in KL divergence + rise in V (valence) right after INSIGHT_CONSOLIDATION.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="RELIEF",
                threshold=0.4,
                window_size=16,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track KL divergence and valence over time
        self.kl_history = []
        self.valence_history = []
        self.insight_consolidation_events = []
        
    def notify_insight_consolidation(self, timestamp: int, intensity: float):
        """Called when INSIGHT_CONSOLIDATION probe fires"""
        self.insight_consolidation_events.append((timestamp, intensity))
        # Keep only recent events
        cutoff = self.token_position - self.config.window_size * 2
        self.insight_consolidation_events = [(t, i) for t, i in self.insight_consolidation_events if t > cutoff]
    
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       guarded_logits: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for relief detection"""
        
        if raw_logits is None or guarded_logits is None:
            return None
        
        # Compute KL divergence
        p = F.softmax(raw_logits, dim=-1)
        q = F.softmax(guarded_logits, dim=-1)
        kl_div = F.kl_div(
            torch.log(q + 1e-8), 
            p, 
            reduction='batchmean'
        ).item()
        
        # Compute valence (simplified: entropy reduction = positive valence)
        raw_probs = F.softmax(raw_logits, dim=-1)
        guarded_probs = F.softmax(guarded_logits, dim=-1)
        raw_entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-8), dim=-1).item()
        guarded_entropy = -torch.sum(guarded_probs * torch.log(guarded_probs + 1e-8), dim=-1).item()
        valence = raw_entropy - guarded_entropy  # Positive = good, negative = bad
        
        # Update histories
        self.kl_history.append(kl_div)
        self.valence_history.append(valence)
        
        if len(self.kl_history) > self.config.window_size:
            self.kl_history.pop(0)
            self.valence_history.pop(0)
        
        # Collect baseline data
        signals = {
            "kl_divergence": kl_div,
            "valence": valence
        }
        self.collect_baseline(signals)
        
        # Detect relief pattern after insight consolidation
        if (self.baseline_ready and 
            len(self.insight_consolidation_events) > 0 and 
            len(self.kl_history) >= 8):
            
            for event_timestamp, event_intensity in self.insight_consolidation_events:
                tokens_since_insight = self.token_position - event_timestamp
                
                # Check if we're in the relief window (1-8 tokens after insight)
                if 1 <= tokens_since_insight <= 8:
                    # Check for KL drop and valence rise
                    if len(self.kl_history) >= tokens_since_insight:
                        kl_before = self.kl_history[-tokens_since_insight-1] if tokens_since_insight < len(self.kl_history) else self.kl_history[0]
                        kl_after = self.kl_history[-1]
                        kl_drop = kl_before - kl_after
                        
                        valence_before = self.valence_history[-tokens_since_insight-1] if tokens_since_insight < len(self.valence_history) else self.valence_history[0]
                        valence_after = self.valence_history[-1]
                        valence_rise = valence_after - valence_before
                        
                        # Fire if KL dropped and valence rose
                        if kl_drop > 0.1 and valence_rise > 0.1:
                            intensity = min(1.0, (kl_drop + valence_rise) / 2.0)
                            
                            self.fire_probe(intensity, signals, {
                                "kl_drop": kl_drop,
                                "valence_rise": valence_rise,
                                "tokens_since_insight": tokens_since_insight,
                                "insight_intensity": event_intensity,
                                "decision_rule": "kl_drop_plus_valence_rise"
                            })
                            
                            # Remove this event as we've processed it
                            self.insight_consolidation_events.remove((event_timestamp, event_intensity))
                            break
        
        return None

class SocialAttunementProbe(BaseProbe):
    """
    SOCIAL_ATTUNEMENT probe
    
    Measures theory-of-mind Δh dot product similar to prosocial; uses it in social contexts.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None, tom_vector: Optional[np.ndarray] = None):
        if config is None:
            config = ProbeConfig(
                name="SOCIAL_ATTUNEMENT",
                threshold=0.4,
                window_size=20,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Theory-of-mind steering vector (would be loaded from actual steering data)
        self.tom_vector = tom_vector
        self.attunement_history = []
        self.social_context_history = []
        
    def process_signals(self,
                       hidden_states=None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for social attunement detection"""
        
        if self.tom_vector is None or hidden_states is None:
            return None
        
        attunement_score = 0.0
        
        # Extract current hidden state
        if isinstance(hidden_states, list):
            # Use mid-layer hidden state
            mid_layer = len(hidden_states) // 2
            current_hidden = hidden_states[mid_layer][0, -1, :].detach().cpu().numpy()
        else:
            mid_layer = hidden_states.shape[0] // 2
            current_hidden = hidden_states[mid_layer, -1, :].detach().cpu().numpy()
        
        # Compute attunement (cosine similarity with ToM vector)
        if len(current_hidden) == len(self.tom_vector):
            dot_product = np.dot(current_hidden, self.tom_vector)
            norm_hidden = np.linalg.norm(current_hidden)
            norm_tom = np.linalg.norm(self.tom_vector)
            
            if norm_hidden > 0 and norm_tom > 0:
                attunement_score = dot_product / (norm_hidden * norm_tom)
        
        # Update history
        self.attunement_history.append(attunement_score)
        if len(self.attunement_history) > self.config.window_size:
            self.attunement_history.pop(0)
        
        # Collect baseline data
        signals = {"attunement_score": attunement_score}
        self.collect_baseline(signals)
        
        # Detect sustained social attunement
        if len(self.attunement_history) >= 10:
            recent_attunement = np.mean(self.attunement_history[-10:])
            attunement_stability = np.std(self.attunement_history[-10:])
            
            # Fire if high attunement with stability (sustained social engagement)
            if (recent_attunement > self.config.threshold and 
                attunement_stability < 0.2):  # Stable pattern
                
                intensity = min(1.0, recent_attunement * (1.0 - attunement_stability))
                
                self.fire_probe(intensity, signals, {
                    "attunement_score": attunement_score,
                    "recent_attunement": recent_attunement,
                    "attunement_stability": attunement_stability,
                    "decision_rule": "sustained_social_attunement"
                })
        
        return None

class AgencyLossProbe(BaseProbe):
    """
    AGENCY_LOSS / DISSOCIATION probe
    
    Detects high FRAGMENTATION + low C (confidence) + high WM_DROP over several windows.
    """
    
    def __init__(self, config: Optional[ProbeConfig] = None):
        if config is None:
            config = ProbeConfig(
                name="AGENCY_LOSS",
                threshold=0.7,
                window_size=24,
                baseline_tokens=300
            )
        super().__init__(config)
        
        # Track multiple behavioral indicators
        self.fragmentation_history = []
        self.confidence_history = []
        self.wm_drop_history = []
        self.agency_loss_windows = []
        
    def notify_fragmentation(self, timestamp: int, intensity: float):
        """Called when FRAGMENTATION probe fires"""
        self.fragmentation_history.append((timestamp, intensity))
        # Keep only recent events
        cutoff = self.token_position - self.config.window_size * 2
        self.fragmentation_history = [(t, i) for t, i in self.fragmentation_history if t > cutoff]
    
    def notify_wm_drop(self, timestamp: int, intensity: float):
        """Called when WORKING_MEMORY_DROP probe fires"""
        self.wm_drop_history.append((timestamp, intensity))
        # Keep only recent events
        cutoff = self.token_position - self.config.window_size * 2
        self.wm_drop_history = [(t, i) for t, i in self.wm_drop_history if t > cutoff]
    
    def process_signals(self,
                       raw_logits: torch.Tensor = None,
                       **kwargs) -> Optional[ProbeEvent]:
        """Process signals for agency loss detection"""
        
        if raw_logits is None:
            return None
        
        # Compute confidence (entropy of raw logits)
        raw_probs = F.softmax(raw_logits, dim=-1)
        confidence = 1.0 - (-torch.sum(raw_probs * torch.log(raw_probs + 1e-8), dim=-1).item()) / np.log(raw_probs.shape[-1])
        
        # Update confidence history
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.config.window_size:
            self.confidence_history.pop(0)
        
        # Collect baseline data
        signals = {"confidence": confidence}
        self.collect_baseline(signals)
        
        # Detect agency loss pattern over multiple windows
        if (self.baseline_ready and 
            len(self.confidence_history) >= 16):
            
            # Check for sustained low confidence
            recent_confidence = self.confidence_history[-16:]
            low_confidence_periods = sum(1 for c in recent_confidence if c < 0.3)
            
            # Check for recent fragmentation events
            recent_fragmentation = [e for e in self.fragmentation_history 
                                  if self.token_position - e[0] <= 16]
            
            # Check for recent WM drop events
            recent_wm_drops = [e for e in self.wm_drop_history 
                              if self.token_position - e[0] <= 16]
            
            # Agency loss: high fragmentation + low confidence + high WM drops
            if (len(recent_fragmentation) >= 3 and  # Multiple fragmentation events
                low_confidence_periods >= 12 and     # Sustained low confidence
                len(recent_wm_drops) >= 2):         # Multiple WM drops
                
                # Calculate composite intensity
                frag_intensity = np.mean([e[1] for e in recent_fragmentation])
                wm_intensity = np.mean([e[1] for e in recent_wm_drops])
                confidence_penalty = 1.0 - np.mean(recent_confidence)
                
                intensity = min(1.0, (frag_intensity + wm_intensity + confidence_penalty) / 3.0)
                
                self.fire_probe(intensity, signals, {
                    "fragmentation_events": len(recent_fragmentation),
                    "avg_fragmentation_intensity": frag_intensity,
                    "wm_drop_events": len(recent_wm_drops),
                    "avg_wm_drop_intensity": wm_intensity,
                    "low_confidence_periods": low_confidence_periods,
                    "avg_confidence": np.mean(recent_confidence),
                    "decision_rule": "high_fragmentation_low_confidence_high_wm_drops"
                })
        
        return None

# Factory functions for new specialized probes
def create_self_inconsistency_tension_probe(threshold: float = 0.6, **kwargs) -> SelfInconsistencyTensionProbe:
    """Factory function to create a SELF_INCONSISTENCY_TENSION probe"""
    config = ProbeConfig(
        name="SELF_INCONSISTENCY_TENSION",
        threshold=threshold,
        **kwargs
    )
    return SelfInconsistencyTensionProbe(config)

def create_goal_threat_probe(threshold: float = 0.5, **kwargs) -> GoalThreatProbe:
    """Factory function to create a GOAL_THREAT probe"""
    config = ProbeConfig(
        name="GOAL_THREAT",
        threshold=threshold,
        **kwargs
    )
    return GoalThreatProbe(config)

def create_relief_probe(threshold: float = 0.4, **kwargs) -> ReliefProbe:
    """Factory function to create a RELIEF probe"""
    config = ProbeConfig(
        name="RELIEF",
        threshold=threshold,
        **kwargs
    )
    return ReliefProbe(config)

def create_social_attunement_probe(threshold: float = 0.4, tom_vector: Optional[np.ndarray] = None, **kwargs) -> SocialAttunementProbe:
    """Factory function to create a SOCIAL_ATTUNEMENT probe"""
    config = ProbeConfig(
        name="SOCIAL_ATTUNEMENT",
        threshold=threshold,
        **kwargs
    )
    return SocialAttunementProbe(config, tom_vector=tom_vector)

def create_agency_loss_probe(threshold: float = 0.7, **kwargs) -> AgencyLossProbe:
    """Factory function to create an AGENCY_LOSS probe"""
    config = ProbeConfig(
        name="AGENCY_LOSS",
        threshold=threshold,
        **kwargs
    )
    return AgencyLossProbe(config)
