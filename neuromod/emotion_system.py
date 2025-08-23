#!/usr/bin/env python3
"""
Emotion System for Neuromodulation Framework

Computes emotional states from probe firings using a mathematical framework
with 7 latent affect axes and 12 discrete emotions.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque
import logging

from .probes import ProbeEvent

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    """Represents the current emotional state"""
    timestamp: int
    token_position: int
    
    # Latent affect axes (clamped to [-1, 1])
    arousal: float
    valence: float
    certainty: float
    openness: float
    integration: float
    sociality: float
    risk_preference: float
    
    # Discrete emotions (probabilities and intensities)
    emotions: Dict[str, Dict[str, float]]  # emotion_name -> {probability, intensity}
    
    # Raw probe statistics for debugging
    probe_stats: Dict[str, Any]


class EmotionSystem:
    """
    Emotion system that computes emotional states from probe firings.
    
    Uses a sliding window approach to compute rolling averages of probe outputs
    and maps them to 7 latent affect axes, which then determine 12 discrete emotions.
    """
    
    def __init__(self, window_size: int = 64):
        """
        Initialize the emotion system.
        
        Args:
            window_size: Number of tokens to use for sliding window averages
        """
        self.window_size = window_size
        
        # Sliding window buffers for probe statistics
        self.surprisal_buffer = deque(maxlen=window_size)
        self.entropy_buffer = deque(maxlen=window_size)
        self.kl_buffer = deque(maxlen=window_size)
        self.lr_attention_buffer = deque(maxlen=window_size)
        self.prosocial_alignment_buffer = deque(maxlen=window_size)
        self.anti_cliche_buffer = deque(maxlen=window_size)
        self.risk_bend_buffer = deque(maxlen=window_size)
        
        # Probe event counters for rate calculations
        self.probe_events = {
            'NOVEL_LINK': deque(maxlen=window_size),
            'INSIGHT_CONSOLIDATION': deque(maxlen=window_size),
            'FIXATION_FLOW': deque(maxlen=window_size),
            'FRAGMENTATION': deque(maxlen=window_size),
            'WORKING_MEMORY_DROP': deque(maxlen=window_size),
            'AVOID_GUARD': deque(maxlen=window_size)
        }
        
        # Current emotional state
        self.current_state: Optional[EmotionState] = None
        self.emotion_history: List[EmotionState] = []
        
        # Seeded weights for emotion calculations (will be replaced with learned weights)
        self._initialize_emotion_weights()
    
    def _initialize_emotion_weights(self):
        """Initialize seeded emotion weights"""
        # Base weights for each emotion (bias terms)
        self.emotion_bases = {
            'joy': 0.0,
            'calm': 0.0,
            'anxiety': 0.0,
            'curiosity': 0.0,
            'awe': 0.0,
            'frustration': 0.0,
            'confusion': 0.0,
            'flow': 0.0,
            'determination': 0.0,
            'empathy': 0.0,
            'confidence': 0.0,
            'boredom': 0.0
        }
        
        # Feature weights for each emotion
        self.emotion_weights = {
            'joy': {'valence': 1.2, 'arousal': 0.4, 'integration': 0.4, 'kl': -0.3},
            'calm': {'valence': 1.0, 'arousal': -0.8, 'integration': 0.6, 'openness': -0.2},
            'anxiety': {'valence': -1.0, 'arousal': 1.0, 'integration': -0.6, 'kl': 0.6},
            'curiosity': {'openness': 0.9, 'arousal': 0.3, 'integration': 0.2, 'risk': -0.2},
            'awe': {'openness': 1.0, 'arousal': 0.5, 'integration': 0.4},
            'frustration': {'kl': 0.8, 'fragmentation': 0.5, 'integration': -0.4},
            'confusion': {'fragmentation': 0.9, 'wm_drop': 0.6, 'integration': -0.5},
            'flow': {'integration': 0.9, 'certainty': 0.3, 'arousal': 0.2},
            'determination': {'certainty': 0.7, 'entropy': 0.5, 'risk': 0.3},
            'empathy': {'sociality': 1.0, 'valence': 0.3, 'kl': -0.2},
            'confidence': {'certainty': 1.0, 'integration': 0.3, 'avoid_guard': -0.3},
            'boredom': {'arousal': -0.8, 'openness': -0.5, 'wm_drop': 0.2}
        }
        
        # Intensity weights (normalized feature vectors)
        self.intensity_weights = {
            'joy': {'valence': 0.8, 'arousal': 0.3, 'integration': 0.3, 'openness': 0.2},
            'calm': {'valence': 0.7, 'integration': 0.5, 'certainty': 0.3},
            'anxiety': {'arousal': 0.6, 'kl': 0.5, 'integration': -0.3},
            'curiosity': {'openness': 0.8, 'arousal': 0.3, 'integration': 0.2},
            'awe': {'openness': 0.7, 'arousal': 0.4, 'integration': 0.4},
            'frustration': {'kl': 0.6, 'fragmentation': 0.4, 'integration': -0.3},
            'confusion': {'fragmentation': 0.7, 'wm_drop': 0.5, 'integration': -0.4},
            'flow': {'integration': 0.8, 'certainty': 0.3, 'arousal': 0.2},
            'determination': {'certainty': 0.7, 'entropy': 0.4, 'risk': 0.3},
            'empathy': {'sociality': 0.8, 'valence': 0.3, 'certainty': 0.2},
            'confidence': {'certainty': 0.8, 'integration': 0.3, 'sociality': 0.2},
            'boredom': {'arousal': -0.6, 'openness': -0.4, 'wm_drop': 0.3}
        }
    
    def update_probe_statistics(self, probe_event: ProbeEvent):
        """
        Update probe statistics when a probe fires.
        
        Args:
            probe_event: The probe event that fired
        """
        probe_name = probe_event.probe_name
        
        if probe_name in self.probe_events:
            self.probe_events[probe_name].append({
                'timestamp': probe_event.timestamp,
                'intensity': probe_event.intensity,
                'metadata': probe_event.metadata
            })
        
        # Extract additional signals from metadata if available
        if probe_event.metadata:
            if 'surprisal' in probe_event.metadata:
                self.surprisal_buffer.append(probe_event.metadata['surprisal'])
            if 'entropy' in probe_event.metadata:
                self.entropy_buffer.append(probe_event.metadata['entropy'])
            if 'kl_divergence' in probe_event.metadata:
                self.kl_buffer.append(probe_event.metadata['kl_divergence'])
            if 'lr_attention' in probe_event.metadata:
                self.lr_attention_buffer.append(probe_event.metadata['lr_attention'])
            if 'prosocial_alignment' in probe_event.metadata:
                self.prosocial_alignment_buffer.append(probe_event.metadata['prosocial_alignment'])
            if 'anti_cliche_gain' in probe_event.metadata:
                self.anti_cliche_buffer.append(probe_event.metadata['anti_cliche_gain'])
            if 'risk_bend_mass' in probe_event.metadata:
                self.risk_bend_buffer.append(probe_event.metadata['risk_bend_mass'])
    
    def update_raw_signals(self, signals: Dict[str, float]):
        """
        Update raw signal buffers directly.
        
        Args:
            signals: Dictionary of signal values
        """
        if 'surprisal' in signals:
            self.surprisal_buffer.append(signals['surprisal'])
        if 'entropy' in signals:
            self.entropy_buffer.append(signals['entropy'])
        if 'kl_divergence' in signals:
            self.kl_buffer.append(signals['kl_divergence'])
        if 'lr_attention' in signals:
            self.lr_attention_buffer.append(signals['lr_attention'])
        if 'prosocial_alignment' in signals:
            self.prosocial_alignment_buffer.append(signals['prosocial_alignment'])
        if 'anti_cliche_gain' in signals:
            self.anti_cliche_buffer.append(signals['anti_cliche_gain'])
        if 'risk_bend_mass' in signals:
            self.risk_bend_buffer.append(signals['risk_bend_mass'])
    
    def _compute_window_average(self, buffer: deque) -> float:
        """Compute average over sliding window"""
        if not buffer:
            return 0.0
        return np.mean(buffer)
    
    def _compute_window_std(self, buffer: deque) -> float:
        """Compute standard deviation over sliding window"""
        if len(buffer) < 2:
            return 0.0
        return np.std(buffer)
    
    def _compute_probe_rate(self, probe_name: str) -> float:
        """Compute firing rate for a specific probe over the window"""
        events = self.probe_events[probe_name]
        if not events:
            return 0.0
        return len(events) / self.window_size
    
    def _compute_flow_time(self) -> float:
        """Compute time spent in flow state (simplified)"""
        # Simplified: count consecutive FIXATION_FLOW events
        flow_events = self.probe_events['FIXATION_FLOW']
        if not flow_events:
            return 0.0
        
        # Count recent consecutive flow events
        consecutive_flow = 0
        for event in reversed(flow_events):
            if event['intensity'] > 0.5:  # High intensity flow
                consecutive_flow += 1
            else:
                break
        
        return min(consecutive_flow / self.window_size, 1.0)
    
    def compute_latent_axes(self) -> Dict[str, float]:
        """
        Compute the 7 latent affect axes using the mathematical framework.
        
        Returns:
            Dictionary of axis values clamped to [-1, 1]
        """
        # Get window averages
        avg_surprisal = self._compute_window_average(self.surprisal_buffer)
        avg_entropy = self._compute_window_average(self.entropy_buffer)
        avg_kl = self._compute_window_average(self.kl_buffer)
        avg_lr = self._compute_window_average(self.lr_attention_buffer)
        avg_prosocial = self._compute_window_average(self.prosocial_alignment_buffer)
        avg_anti_cliche = self._compute_window_average(self.anti_cliche_buffer)
        avg_risk_bend = self._compute_window_average(self.risk_bend_buffer)
        
        # Get probe rates
        novel_link_rate = self._compute_probe_rate('NOVEL_LINK')
        insight_rate = self._compute_probe_rate('INSIGHT_CONSOLIDATION')
        flow_time = self._compute_flow_time()
        frag_rate = self._compute_probe_rate('FRAGMENTATION')
        wm_drop_rate = self._compute_probe_rate('WORKING_MEMORY_DROP')
        
        # Debug output
        print(f"🔍 Emotion computation debug:")
        print(f"   Buffer sizes: surprisal={len(self.surprisal_buffer)}, entropy={len(self.entropy_buffer)}")
        print(f"   Window averages: surprisal={avg_surprisal:.3f}, entropy={avg_entropy:.3f}")
        print(f"   Probe rates: novel_link={novel_link_rate:.3f}, insight={insight_rate:.3f}")
        
        # Compute axes according to the mathematical framework
        
        # Arousal (A)
        arousal = (0.40 * self._compute_window_std(self.surprisal_buffer) + 
                   0.30 * novel_link_rate + 
                   0.20 * (1.0 - avg_entropy) + 
                   0.10 * avg_lr)
        
        # Valence (V)
        valence = (0.60 * avg_prosocial - 
                   0.20 * avg_kl - 
                   0.20 * self._compute_window_std(self.entropy_buffer))
        
        # Certainty/Agency (C)
        certainty = (-0.70 * avg_entropy - 
                     0.30 * avg_kl)
        
        # Openness/Novelty (N)
        openness = (0.60 * novel_link_rate + 
                    0.20 * avg_anti_cliche + 
                    0.20 * avg_anti_cliche)  # Simplified: using same value twice
        
        # Integration/Coherence (G)
        integration = (0.45 * flow_time - 
                       0.35 * frag_rate - 
                       0.30 * wm_drop_rate + 
                       0.20 * insight_rate)
        
        # Sociality/Warmth (S)
        sociality = avg_prosocial
        
        # Risk Preference (R)
        risk_preference = -avg_risk_bend
        
        # Debug raw values before clamping
        print(f"   Raw values: arousal={arousal:.3f}, valence={valence:.3f}, certainty={certainty:.3f}")
        
        # Clamp all axes to [-1, 1] using tanh
        axes = {
            'arousal': np.tanh(arousal),
            'valence': np.tanh(valence),
            'certainty': np.tanh(certainty),
            'openness': np.tanh(openness),
            'integration': np.tanh(integration),
            'sociality': np.tanh(sociality),
            'risk_preference': np.tanh(risk_preference)
        }
        
        print(f"   Final axes: arousal={axes['arousal']:.3f}, valence={axes['valence']:.3f}, certainty={axes['certainty']:.3f}")
        
        return axes
    
    def compute_discrete_emotions(self, axes: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Compute 12 discrete emotions from latent axes.
        
        Args:
            axes: Dictionary of latent affect axes
            
        Returns:
            Dictionary of emotions with probability and intensity
        """
        emotions = {}
        
        # Get additional probe rates for emotion calculations
        kl_rate = self._compute_probe_rate('AVOID_GUARD')  # Simplified proxy
        frag_rate = self._compute_probe_rate('FRAGMENTATION')
        wm_drop_rate = self._compute_probe_rate('WORKING_MEMORY_DROP')
        avoid_guard_rate = self._compute_probe_rate('AVOID_GUARD')
        
        # Compute each emotion according to the seeded mappings
        for emotion_name, weights in self.emotion_weights.items():
            # Compute logit
            logit = self.emotion_bases[emotion_name]
            
            for feature, weight in weights.items():
                if feature in axes:
                    logit += weight * axes[feature]
                elif feature == 'kl':
                    logit += weight * kl_rate
                elif feature == 'fragmentation':
                    logit += weight * frag_rate
                elif feature == 'wm_drop':
                    logit += weight * wm_drop_rate
                elif feature == 'avoid_guard':
                    logit += weight * avoid_guard_rate
                elif feature == 'entropy':
                    logit += weight * (1.0 - axes.get('certainty', 0.0))  # Simplified proxy
            
            # Compute probability using sigmoid
            probability = 1.0 / (1.0 + np.exp(-logit))
            
            # Compute intensity using intensity weights
            intensity = 0.0
            intensity_weights = self.intensity_weights[emotion_name]
            
            for feature, weight in intensity_weights.items():
                if feature in axes:
                    intensity += weight * axes[feature]
                elif feature == 'kl':
                    intensity += weight * kl_rate
                elif feature == 'fragmentation':
                    intensity += weight * frag_rate
                elif feature == 'wm_drop':
                    intensity += weight * wm_drop_rate
                elif feature == 'avoid_guard':
                    intensity += weight * avoid_guard_rate
                elif feature == 'entropy':
                    intensity += weight * (1.0 - axes.get('certainty', 0.0))
            
            # Normalize intensity to [0, 1]
            intensity = np.clip(intensity, 0.0, 1.0)
            
            emotions[emotion_name] = {
                'probability': probability,
                'intensity': intensity
            }
        
        return emotions
    
    def update_emotion_state(self, token_position: int) -> EmotionState:
        """
        Update the current emotional state based on current probe statistics.
        
        Args:
            token_position: Current token position
            
        Returns:
            Updated emotion state
        """
        # Compute latent axes
        axes = self.compute_latent_axes()
        
        # Compute discrete emotions
        emotions = self.compute_discrete_emotions(axes)
        
        # Create emotion state
        state = EmotionState(
            timestamp=token_position,
            token_position=token_position,
            arousal=axes['arousal'],
            valence=axes['valence'],
            certainty=axes['certainty'],
            openness=axes['openness'],
            integration=axes['integration'],
            sociality=axes['sociality'],
            risk_preference=axes['risk_preference'],
            emotions=emotions,
            probe_stats={
                'surprisal_buffer_size': len(self.surprisal_buffer),
                'entropy_buffer_size': len(self.entropy_buffer),
                'kl_buffer_size': len(self.kl_buffer),
                'probe_events': {name: len(events) for name, events in self.probe_events.items()}
            }
        )
        
        # Update current state and history
        self.current_state = state
        self.emotion_history.append(state)
        
        return state
    
    def get_current_emotion_state(self) -> Optional[EmotionState]:
        """Get the current emotional state"""
        return self.current_state
    
    def get_emotion_history(self, window_size: Optional[int] = None) -> List[EmotionState]:
        """
        Get emotion history, optionally limited to recent states.
        
        Args:
            window_size: Number of recent states to return (None for all)
            
        Returns:
            List of emotion states
        """
        if window_size is None:
            return self.emotion_history
        
        return self.emotion_history[-window_size:]
    
    def get_dominant_emotions(self, top_k: int = 3) -> List[Tuple[str, float, float]]:
        """
        Get the top-k dominant emotions by probability.
        
        Args:
            top_k: Number of top emotions to return
            
        Returns:
            List of (emotion_name, probability, intensity) tuples
        """
        if not self.current_state:
            return []
        
        emotions = self.current_state.emotions
        sorted_emotions = sorted(
            emotions.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        return [(name, data['probability'], data['intensity']) 
                for name, data in sorted_emotions[:top_k]]
    
    def reset(self):
        """Reset the emotion system state"""
        # Clear all buffers
        self.surprisal_buffer.clear()
        self.entropy_buffer.clear()
        self.kl_buffer.clear()
        self.lr_attention_buffer.clear()
        self.prosocial_alignment_buffer.clear()
        self.anti_cliche_buffer.clear()
        self.risk_bend_buffer.clear()
        
        # Clear probe events
        for events in self.probe_events.values():
            events.clear()
        
        # Clear emotion history
        self.current_state = None
        self.emotion_history.clear()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and statistics"""
        return {
            'window_size': self.window_size,
            'buffer_sizes': {
                'surprisal': len(self.surprisal_buffer),
                'entropy': len(self.entropy_buffer),
                'kl': len(self.kl_buffer),
                'lr_attention': len(self.lr_attention_buffer),
                'prosocial': len(self.prosocial_alignment_buffer),
                'anti_cliche': len(self.anti_cliche_buffer),
                'risk_bend': len(self.risk_bend_buffer)
            },
            'probe_rates': {
                name: len(events) / self.window_size 
                for name, events in self.probe_events.items()
            },
            'emotion_history_size': len(self.emotion_history),
            'current_state': self.current_state is not None
        }
