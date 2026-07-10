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
from ..probes import ProbeBus, ProbeEvent, create_jlens_probe
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
    # Grounded J-lens workspace telemetry: {"workspace_occupancy": float,
    # "workspace_<concept>": float, ...}. Empty when no J-space basis is supplied.
    workspace: Dict[str, float] = None

    def __post_init__(self):
        if self.workspace is None:
            self.workspace = {}

class ProbeEvaluator:
    """
    Evaluates model outputs using the real probe and emotion system.
    
    This replaces the simple keyword-based evaluation with actual
    probe signal processing and emotion computation.
    """
    
    def __init__(self, model_manager: ModelSupportManager = None, jspace_basis=None):
        """
        Args:
            model_manager: Model support manager.
            jspace_basis: Optional fitted :class:`neuromod.jspace.JSpaceBasis`. When
                provided, real J-lens workspace telemetry (occupancy + per-concept
                scores) is read from the model's hidden states each token and fed to
                the emotion system -- replacing the previously *simulated*
                (np.random) internal-state signals.
        """
        self.model_manager = model_manager or ModelSupportManager(test_mode=True)
        self.emotion_system = EmotionSystem()
        self.pack_registry = PackRegistry()
        self.jspace_basis = jspace_basis
        
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
            # Get tokenizer if available
            tokenizer = getattr(model, 'tokenizer', None)
            if tokenizer is None and hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer
            pack_manager.apply_pack(pack, model, tokenizer=tokenizer)
        except ValueError:
            logger.warning(f"Pack '{pack_name}' not found, using base model")
        
        # Initialize probe bus for this evaluation
        probe_bus = ProbeBus()
        # If we have a J-space basis, register the J-lens telemetry probe so the
        # workspace is read directly from hidden states each token.
        if self.jspace_basis is not None:
            probe_bus.register_probe(create_jlens_probe(self.jspace_basis))

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

        # Grounded J-lens workspace telemetry, exposed as workspace_<concept> keys so
        # WORKSPACE_CONCEPT targets can be scored against real internal state.
        workspace = {}
        if self.jspace_basis is not None:
            workspace["workspace_occupancy"] = self.emotion_system.get_workspace_occupancy()
            for concept, val in self.emotion_system.get_workspace_concepts().items():
                workspace[f"workspace_{concept}"] = val

        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")

        return ProbeEvaluationResult(
            emotions=avg_emotions,
            latent_axes=avg_axes,
            probe_stats=avg_probe_stats,
            text_metrics=text_metrics,
            overall_score=overall_score,
            workspace=workspace
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
            
            want_hidden = self.jspace_basis is not None
            with torch.no_grad():
                for token_pos in range(50):  # Max 50 new tokens
                    global_token_pos = prompt_idx * 100 + token_pos  # Unique position across all prompts
                    # Forward pass (request hidden states when we have a J-space basis
                    # so we can read real workspace telemetry from them).
                    outputs = model(generated_ids, output_hidden_states=want_hidden)
                    logits = outputs.logits
                    hidden_states = getattr(outputs, "hidden_states", None)

                    # Get next token probabilities
                    next_token_logits = logits[0, -1, :]
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Sample next token
                    next_token_id = torch.multinomial(probs, 1).item()

                    # Stop if EOS token
                    if next_token_id == tokenizer.eos_token_id:
                        break

                    # Real, measurable signals only. (The previous version fabricated
                    # kl/lr_attention/prosocial/anti_cliche/risk with np.random.normal,
                    # so the optimizer was tuning packs against noise. We now pass only
                    # signals we can actually compute: entropy/surprisal from the logits,
                    # and -- when a J-space basis is available -- token-grounded workspace
                    # telemetry from the hidden states. Unmeasured channels are omitted,
                    # not faked; their buffers simply stay empty.)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
                    surprisal = -torch.log(torch.tensor(probs[next_token_id].item() + 1e-8)).item()
                    signals = {'entropy': entropy, 'surprisal': surprisal}

                    if hidden_states is not None:
                        signals.update(self._jlens_signals(hidden_states))

                    self.emotion_system.update_raw_signals(signals)

                    # Process probe signals (pass hidden_states so a registered
                    # JLensProbe can read the workspace directly).
                    probe_bus.process_signals(hidden_states=hidden_states, **signals)

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
    
    def _jlens_signals(self, hidden_states) -> Dict[str, float]:
        """Read grounded workspace telemetry from hidden states via the J-space basis.

        Returns a ``workspace_occupancy`` scalar and a ``concept::<name>`` score per
        basis concept, at the deepest fitted workspace layer, for the last token.
        """
        basis = self.jspace_basis
        try:
            jl = basis.layer_indices[-1]
            if isinstance(hidden_states, (list, tuple)):
                hs = hidden_states[jl] if jl < len(hidden_states) else hidden_states[-1]
            else:
                hs = hidden_states
            if hs.dim() == 3:
                h = hs[0, -1, :]
            elif hs.dim() == 2:
                h = hs[-1, :]
            else:
                h = hs
            h = h.detach().to(torch.float32).cpu()
            signals = {"workspace_occupancy": float(basis.occupancy(h, layer=jl))}
            for concept, score in basis.readout(h, layer=jl):
                signals[f"concept::{concept}"] = float(score)
            return signals
        except Exception as e:
            logger.debug(f"J-lens signal extraction failed: {e}")
            return {}

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
