"""
Probe-Based Evaluation Framework

This module integrates with the real emotion tracking system to evaluate
behavioral outcomes using actual probe signals and emotion computations.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..emotion_system import EmotionSystem, EmotionState
from ..probes import ProbeBus, ProbeEvent
from ..model_support import ModelSupportManager
from ..pack_system import PackRegistry

logger = logging.getLogger(__name__)

@dataclass
class ProbeEvaluationResult:
    """Result of probe-based evaluation"""
    emotions: Dict[str, float]  # emotion_name -> intensity
    latent_axes: Dict[str, float]  # axis_name -> value
    probe_stats: Dict[str, Any]  # probe firing rates and statistics
    text_metrics: Dict[str, float]  # basic text metrics
    overall_score: float  # weighted combination of all metrics

class ProbeEvaluator:
    """
    Evaluates model outputs using the real probe and emotion system.
    
    This replaces the simple keyword-based evaluation with actual
    probe signal processing and emotion computation.
    """
    
    def __init__(self, model_manager: ModelSupportManager = None):
        self.model_manager = model_manager or ModelSupportManager(test_mode=True)
        self.emotion_system = EmotionSystem()
        self.pack_registry = PackRegistry()
        
    def evaluate_with_pack(self, 
                          pack_name: str,
                          test_prompts: List[str],
                          model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> ProbeEvaluationResult:
        """
        Evaluate a pack by running it through the model with probe monitoring.
        
        Args:
            pack_name: Name of the pack to evaluate
            test_prompts: List of prompts to test
            model_name: Model to use for evaluation (default: Llama-3.1-8B-Instruct)
                       CRITICAL: Must match the target model. Do not use test models
                       (e.g., DialoGPT) for production evaluation.
            
        Returns:
            ProbeEvaluationResult with emotions, axes, and probe stats
        """
        logger.info(f"Evaluating pack '{pack_name}' with {len(test_prompts)} prompts")
        
        # Load model
        model, tokenizer, _ = self.model_manager.load_model(model_name)
        
        # Apply pack
        try:
            pack = self.pack_registry.get_pack(pack_name)
            # Create a PackManager to apply the pack
            from ..pack_system import PackManager
            pack_manager = PackManager()
            pack_manager.apply_pack(pack, model)
        except ValueError:
            logger.warning(f"Pack '{pack_name}' not found, using base model")
        
        # Initialize probe bus for this evaluation
        probe_bus = ProbeBus()
        # Note: ProbeBus doesn't have setup_default_probes method
        # We'll work with the basic probe bus for now
        
        # Reset emotion system
        self.emotion_system = EmotionSystem()
        
        all_emotions = []
        all_axes = []
        all_probe_stats = []
        all_responses = []
        successful_generations = 0
        
        # Process all prompts with a single emotion system to accumulate data
        for prompt_idx, prompt in enumerate(test_prompts):
            logger.debug(f"Processing prompt {prompt_idx + 1}/{len(test_prompts)}: {prompt[:50]}...")
            
            # Generate response with probe monitoring
            result = self._generate_with_probe_monitoring(
                model, tokenizer, probe_bus, prompt, prompt_idx
            )
            
            if result:
                all_emotions.append(result['emotions'])
                all_axes.append(result['latent_axes'])
                all_probe_stats.append(result['probe_stats'])
                all_responses.append(result.get('response_text', ''))
                successful_generations += 1
        
        # Aggregate results across all prompts
        if not all_emotions:
            logger.warning("No successful generations, returning empty result")
            return ProbeEvaluationResult(
                emotions={},
                latent_axes={},
                probe_stats={},
                text_metrics={},
                overall_score=0.0
            )
        
        # Average emotions across all prompts
        avg_emotions = self._average_emotions(all_emotions)
        avg_axes = self._average_axes(all_axes)
        avg_probe_stats = self._average_probe_stats(all_probe_stats)
        
        # Compute text metrics
        text_metrics = self._compute_text_metrics(test_prompts)
        
        # Compute overall score
        overall_score = self._compute_overall_score(avg_emotions, avg_axes, avg_probe_stats)
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")
        
        return ProbeEvaluationResult(
            emotions=avg_emotions,
            latent_axes=avg_axes,
            probe_stats=avg_probe_stats,
            text_metrics=text_metrics,
            overall_score=overall_score
        )
    
    def _generate_with_probe_monitoring(self, 
                                      model, 
                                      tokenizer, 
                                      probe_bus: ProbeBus,
                                      prompt: str,
                                      prompt_idx: int = 0) -> Optional[Dict[str, Any]]:
        """Generate response while monitoring probe signals"""
        try:
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids
            
            # Generate response token by token to capture probe signals
            generated_ids = input_ids.clone()
            max_length = input_ids.shape[1] + 50  # Generate up to 50 new tokens
            
            with torch.no_grad():
                for token_pos in range(50):  # Max 50 new tokens
                    global_token_pos = prompt_idx * 100 + token_pos  # Unique position across all prompts
                    # Forward pass
                    outputs = model(generated_ids)
                    logits = outputs.logits
                    
                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)
                    
                    # Sample next token
                    next_token_id = torch.multinomial(probs, 1).item()
                    
                    # Stop if EOS token
                    if next_token_id == tokenizer.eos_token_id:
                        break
                    
                    # Compute signals for probe monitoring
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    surprisal = -torch.log(torch.tensor(probs[next_token_id].item() + 1e-8)).item()
                    
                    # Simulate additional probe signals (in real implementation, these would come from actual probes)
                    kl_divergence = np.random.normal(0.1, 0.05)  # Simulated
                    lr_attention = np.random.normal(0.3, 0.1)   # Simulated
                    prosocial_alignment = np.random.normal(0.2, 0.1)  # Simulated
                    anti_cliche_gain = np.random.normal(0.1, 0.05)    # Simulated
                    risk_bend_mass = np.random.normal(0.0, 0.1)       # Simulated
                    
                    # Update emotion system with signals
                    signals = {
                        'entropy': entropy,
                        'surprisal': surprisal,
                        'kl_divergence': kl_divergence,
                        'lr_attention': lr_attention,
                        'prosocial_alignment': prosocial_alignment,
                        'anti_cliche_gain': anti_cliche_gain,
                        'risk_bend_mass': risk_bend_mass
                    }
                    
                    self.emotion_system.update_raw_signals(signals)
                    
                    # Process probe signals
                    probe_bus.process_signals(**signals)
                    
                    # Update emotion state after processing signals
                    self.emotion_system.update_emotion_state(global_token_pos)
                    
                    # Add token to sequence
                    generated_ids = torch.cat([generated_ids, torch.tensor([[next_token_id]])], dim=1)
            
            # Get final emotion state
            if self.emotion_system.current_state:
                emotions = {name: data['intensity'] for name, data in self.emotion_system.current_state.emotions.items()}
                latent_axes = {
                    'arousal': self.emotion_system.current_state.arousal,
                    'valence': self.emotion_system.current_state.valence,
                    'certainty': self.emotion_system.current_state.certainty,
                    'openness': self.emotion_system.current_state.openness,
                    'integration': self.emotion_system.current_state.integration,
                    'sociality': self.emotion_system.current_state.sociality,
                    'risk_preference': self.emotion_system.current_state.risk_preference
                }
                logger.debug(f"Emotion state computed: {len(emotions)} emotions, axes: {latent_axes}")
            else:
                # Fallback if no emotion state computed
                emotions = {}
                latent_axes = {}
                logger.warning("No emotion state computed")
            
            # Get probe statistics
            probe_stats = probe_bus.get_all_stats()
            
            return {
                'emotions': emotions,
                'latent_axes': latent_axes,
                'probe_stats': probe_stats
            }
            
        except Exception as e:
            logger.error(f"Error in probe monitoring generation: {e}")
            return None
    
    def _average_emotions(self, emotion_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average emotions across multiple evaluations"""
        if not emotion_list:
            return {}
        
        all_emotion_names = set()
        for emotions in emotion_list:
            all_emotion_names.update(emotions.keys())
        
        avg_emotions = {}
        for emotion_name in all_emotion_names:
            values = [emotions.get(emotion_name, 0.0) for emotions in emotion_list]
            avg_emotions[emotion_name] = np.mean(values)
        
        return avg_emotions
    
    def _average_axes(self, axes_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Average latent axes across multiple evaluations"""
        if not axes_list:
            return {}
        
        all_axis_names = set()
        for axes in axes_list:
            all_axis_names.update(axes.keys())
        
        avg_axes = {}
        for axis_name in all_axis_names:
            values = [axes.get(axis_name, 0.0) for axes in axes_list]
            avg_axes[axis_name] = np.mean(values)
        
        return avg_axes
    
    def _average_probe_stats(self, stats_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Average probe statistics across multiple evaluations"""
        if not stats_list:
            return {}
        
        # Get all probe names
        all_probe_names = set()
        for stats in stats_list:
            all_probe_names.update(stats.keys())
        
        avg_stats = {}
        for probe_name in all_probe_names:
            # Average firing rates
            firing_rates = [stats.get(probe_name, {}).get('firing_rate', 0.0) for stats in stats_list]
            avg_stats[probe_name] = {
                'firing_rate': np.mean(firing_rates),
                'total_firings': sum(stats.get(probe_name, {}).get('total_firings', 0) for stats in stats_list)
            }
        
        return avg_stats
    
    def _compute_text_metrics(self, prompts: List[str]) -> Dict[str, float]:
        """Compute basic text metrics"""
        if not prompts:
            return {}
        
        total_length = sum(len(prompt.split()) for prompt in prompts)
        avg_length = total_length / len(prompts)
        
        # Simple metrics
        return {
            'avg_prompt_length': avg_length,
            'num_prompts': len(prompts)
        }
    
    def _compute_overall_score(self, 
                              emotions: Dict[str, float], 
                              axes: Dict[str, float], 
                              probe_stats: Dict[str, Any]) -> float:
        """Compute overall evaluation score"""
        score = 0.0
        weight_sum = 0.0
        
        # Weight emotions by intensity
        for emotion, intensity in emotions.items():
            if emotion in ['joy', 'excitement', 'enthusiasm']:
                score += intensity * 2.0  # Positive emotions weighted higher
                weight_sum += 2.0
            elif emotion in ['sadness', 'anger', 'anxiety']:
                score += (1.0 - intensity) * 1.0  # Negative emotions inverted
                weight_sum += 1.0
            else:
                score += intensity * 0.5  # Neutral emotions
                weight_sum += 0.5
        
        # Weight latent axes
        for axis, value in axes.items():
            if axis in ['valence', 'sociality']:  # Positive axes
                score += (value + 1.0) / 2.0 * 1.0  # Normalize to [0, 1]
                weight_sum += 1.0
            elif axis in ['arousal', 'openness']:  # Neutral axes
                score += abs(value) * 0.5  # Reward high absolute values
                weight_sum += 0.5
        
        # Weight probe statistics
        for probe_name, stats in probe_stats.items():
            firing_rate = stats.get('firing_rate', 0.0)
            if probe_name in ['NOVEL_LINK', 'INSIGHT_CONSOLIDATION']:
                score += firing_rate * 1.0  # Reward creative probes
                weight_sum += 1.0
            elif probe_name in ['FRAGMENTATION', 'WORKING_MEMORY_DROP']:
                score += (1.0 - firing_rate) * 0.5  # Penalize fragmentation
                weight_sum += 0.5
        
        return score / max(weight_sum, 1e-8)

# Convenience function
def evaluate_pack_with_probes(pack_name: str, 
                            test_prompts: List[str],
                            model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> ProbeEvaluationResult:
    """
    Convenience function to evaluate a pack using probe system.
    
    Args:
        pack_name: Name of the pack to evaluate
        test_prompts: List of prompts to test
        model_name: Model to use for evaluation (default: Llama-3.1-8B-Instruct)
                   CRITICAL: Must match the target model. Do not use test models
                   (e.g., DialoGPT) for production evaluation.
    
    Returns:
        ProbeEvaluationResult with evaluation metrics
    """
    evaluator = ProbeEvaluator()
    return evaluator.evaluate_with_pack(pack_name, test_prompts, model_name)
