"""
Pack Optimization Engine

This module provides the core optimization engine for tuning pack parameters
to achieve specific behavioral targets.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import copy
from pathlib import Path

from .targets import BehavioralTarget, TargetManager
from .evaluation import EvaluationFramework, BehavioralMetrics
from .probe_evaluator import ProbeEvaluator, ProbeEvaluationResult
from .bayesian_optimizer import BayesianOptimizer, BayesianOptimizationConfig
from .rl_optimizer import RLOptimizer, RLOptimizationConfig
from .evolutionary_ops import PackMutator, PackCrossover, MutationConfig, CrossoverConfig
from ..pack_system import Pack, PackManager, PackRegistry
from ..model_support import ModelSupportManager

logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Available optimization methods"""
    GRADIENT_DESCENT = "gradient_descent"
    EVOLUTIONARY = "evolutionary"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    REINFORCEMENT_LEARNING = "reinforcement_learning"

@dataclass
class OptimizationConfig:
    """Configuration for optimization"""
    method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT
    max_iterations: int = 100
    learning_rate: float = 0.01
    population_size: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    early_stopping_patience: int = 10
    convergence_threshold: float = 1e-4
    validation_split: float = 0.2
    random_seed: Optional[int] = None
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Target model for optimization

@dataclass
class OptimizationResult:
    """Result of pack optimization"""
    optimized_pack: Pack
    final_loss: float
    iteration_history: List[float]
    best_loss: float
    convergence_iteration: int
    success: bool
    iterations: int
    converged: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class PackOptimizer:
    """Main pack optimization engine"""
    
    def __init__(self, 
                 model_manager: ModelSupportManager,
                 evaluation_framework: EvaluationFramework = None,
                 config: OptimizationConfig = None):
        """
        Initialize pack optimizer.
        
        Args:
            model_manager: Model support manager for loading models
            evaluation_framework: Framework for evaluating pack effects
            config: Optimization configuration (model_name should be set in config)
                   CRITICAL: model_name in config must match the model used for final evaluation.
                   Do not use test models (e.g., DialoGPT) for production optimization.
        """
        self.model_manager = model_manager
        self.evaluation_framework = evaluation_framework or EvaluationFramework()
        self.probe_evaluator = ProbeEvaluator(model_manager)
        self.config = config or OptimizationConfig()
        self.pack_registry = PackRegistry()
        # Get model_name from config (removed hardcoded DialoGPT)
        self.model_name = self.config.model_name
        
        # Initialize evolutionary operators
        self.mutator = PackMutator(MutationConfig(mutation_rate=self.config.mutation_rate))
        self.crossover = PackCrossover(CrossoverConfig(crossover_rate=self.config.crossover_rate))
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
        
        logger.info(f"PackOptimizer initialized with target model: {self.model_name}")
    
    def optimize_pack(self, 
                     pack: Pack, 
                     target: BehavioralTarget,
                     test_prompts: List[str] = None) -> OptimizationResult:
        """Optimize a pack to achieve behavioral target"""
        
        if test_prompts is None:
            test_prompts = target.test_prompts
        
        if not test_prompts:
            raise ValueError("No test prompts provided for optimization")
        
        logger.info(f"Starting pack optimization for target: {target.name}")
        logger.info(f"Method: {self.config.method.value}")
        logger.info(f"Test prompts: {len(test_prompts)}")
        
        # Create a copy of the pack for optimization
        optimized_pack = copy.deepcopy(pack)
        
        if self.config.method == OptimizationMethod.GRADIENT_DESCENT:
            return self._optimize_gradient_descent(optimized_pack, target, test_prompts)
        elif self.config.method == OptimizationMethod.EVOLUTIONARY:
            return self._optimize_evolutionary(optimized_pack, target, test_prompts)
        elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
            return self._optimize_random_search(optimized_pack, target, test_prompts)
        elif self.config.method == OptimizationMethod.BAYESIAN:
            return self._optimize_bayesian(optimized_pack, target, test_prompts)
        elif self.config.method == OptimizationMethod.REINFORCEMENT_LEARNING:
            return self._optimize_reinforcement_learning(optimized_pack, target, test_prompts)
        else:
            raise ValueError(f"Unknown optimization method: {self.config.method}")
    
    def _optimize_gradient_descent(self, 
                                  pack: Pack, 
                                  target: BehavioralTarget,
                                  test_prompts: List[str]) -> OptimizationResult:
        """Optimize using gradient descent (if pack parameters are differentiable)"""
        
        # For now, implement a simplified version that doesn't require gradients
        # In a full implementation, we'd need to make pack parameters differentiable
        logger.warning("Gradient descent not fully implemented - falling back to random search")
        return self._optimize_random_search(pack, target, test_prompts)
    
    def _optimize_evolutionary(self, 
                              pack: Pack, 
                              target: BehavioralTarget,
                              test_prompts: List[str]) -> OptimizationResult:
        """Optimize using evolutionary algorithms"""
        
        logger.info("Starting evolutionary optimization")
        
        # Initialize population
        population = self._initialize_population(pack, self.config.population_size)
        best_pack = None
        best_loss = float('inf')
        iteration_history = []
        
        for iteration in range(self.config.max_iterations):
            # Evaluate population
            fitness_scores = []
            for individual in population:
                loss = self._evaluate_pack(individual, target, test_prompts)
                fitness_scores.append(loss)
                
                if loss < best_loss:
                    best_loss = loss
                    best_pack = copy.deepcopy(individual)
            
            iteration_history.append(best_loss)
            
            # Check convergence
            if self._check_convergence(iteration_history):
                break
            
            # Create new generation
            new_population = []
            
            # Keep best individuals (elitism)
            elite_size = max(1, self.config.population_size // 10)
            elite_indices = np.argsort(fitness_scores)[:elite_size]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(population[idx]))
            
            # Generate offspring
            while len(new_population) < self.config.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < self.config.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Mutation
                if np.random.random() < self.config.mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < self.config.mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.config.population_size]
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best loss = {best_loss:.4f}")
        
        return OptimizationResult(
            optimized_pack=best_pack or pack,
            final_loss=best_loss,
            iteration_history=iteration_history,
            best_loss=best_loss,
            convergence_iteration=len(iteration_history),
            success=best_loss < float('inf'),
            iterations=len(iteration_history),
            converged=best_loss < float('inf'),
            metadata={'method': 'evolutionary'}
        )
    
    def _optimize_random_search(self, 
                               pack: Pack, 
                               target: BehavioralTarget,
                               test_prompts: List[str]) -> OptimizationResult:
        """Optimize using random search"""
        
        logger.info("Starting random search optimization")
        
        best_pack = copy.deepcopy(pack)
        best_loss = self._evaluate_pack(best_pack, target, test_prompts)
        iteration_history = [best_loss]
        
        for iteration in range(self.config.max_iterations):
            # Create random variant
            candidate_pack = self._mutate(copy.deepcopy(best_pack))
            
            # Evaluate candidate
            candidate_loss = self._evaluate_pack(candidate_pack, target, test_prompts)
            
            # Update best if better
            if candidate_loss < best_loss:
                best_pack = candidate_pack
                best_loss = candidate_loss
            
            iteration_history.append(best_loss)
            
            # Check convergence
            if self._check_convergence(iteration_history):
                break
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best loss = {best_loss:.4f}")
        
        return OptimizationResult(
            optimized_pack=best_pack,
            final_loss=best_loss,
            iteration_history=iteration_history,
            best_loss=best_loss,
            convergence_iteration=len(iteration_history),
            success=best_loss < float('inf'),
            iterations=len(iteration_history),
            converged=best_loss < float('inf'),
            metadata={'method': 'random_search'}
        )
    
    def _optimize_bayesian(self, 
                          pack: Pack, 
                          target: BehavioralTarget,
                          test_prompts: List[str]) -> OptimizationResult:
        """Optimize using Bayesian optimization"""
        
        logger.info("Starting Bayesian optimization")
        
        # Define parameter bounds (simplified - in practice we'd need to know pack structure)
        # For now, we'll use a simplified approach with effect parameters
        bounds = self._get_pack_parameter_bounds(pack)
        
        # Create objective function
        def objective(params):
            # Create pack with given parameters
            test_pack = self._create_pack_from_parameters(pack, params)
            return self._evaluate_pack(test_pack, target, test_prompts)
        
        # Run Bayesian optimization
        bayesian_config = BayesianOptimizationConfig(
            n_initial_points=min(10, self.config.max_iterations // 5),
            n_iterations=self.config.max_iterations,
            random_seed=self.config.random_seed
        )
        
        optimizer = BayesianOptimizer(bayesian_config)
        best_params, best_loss, history = optimizer.optimize(objective, bounds)
        
        # Create optimized pack
        optimized_pack = self._create_pack_from_parameters(pack, best_params)
        
        return OptimizationResult(
            optimized_pack=optimized_pack,
            final_loss=best_loss,
            iteration_history=history,
            best_loss=best_loss,
            convergence_iteration=len(history),
            success=best_loss < float('inf'),
            iterations=len(history),
            converged=best_loss < float('inf'),
            metadata={'method': 'bayesian'}
        )
    
    def _optimize_reinforcement_learning(self, 
                                       pack: Pack, 
                                       target: BehavioralTarget,
                                       test_prompts: List[str]) -> OptimizationResult:
        """Optimize using reinforcement learning"""
        
        logger.info("Starting reinforcement learning optimization")
        
        # Define parameter bounds
        bounds = self._get_pack_parameter_bounds(pack)
        
        # Create objective function (negate for reward)
        def objective(params):
            test_pack = self._create_pack_from_parameters(pack, params)
            loss = self._evaluate_pack(test_pack, target, test_prompts)
            return loss  # RL optimizer will negate this for reward
        
        # Run RL optimization
        rl_config = RLOptimizationConfig(
            n_episodes=self.config.max_iterations,
            random_seed=self.config.random_seed
        )
        
        optimizer = RLOptimizer(rl_config)
        best_params, best_loss, history = optimizer.optimize(objective, bounds)
        
        # Create optimized pack
        optimized_pack = self._create_pack_from_parameters(pack, best_params)
        
        return OptimizationResult(
            optimized_pack=optimized_pack,
            final_loss=best_loss,
            iteration_history=history,
            best_loss=best_loss,
            convergence_iteration=len(history),
            success=best_loss < float('inf'),
            iterations=len(history),
            converged=best_loss < float('inf'),
            metadata={'method': 'reinforcement_learning'}
        )
    
    def _initialize_population(self, base_pack: Pack, size: int) -> List[Pack]:
        """Initialize population for evolutionary algorithms"""
        population = []
        
        # Add original pack
        population.append(copy.deepcopy(base_pack))
        
        # Generate random variants
        for _ in range(size - 1):
            variant = self._mutate(copy.deepcopy(base_pack))
            population.append(variant)
        
        return population
    
    def _evaluate_pack(self, pack: Pack, target: BehavioralTarget, test_prompts: List[str]) -> float:
        """Evaluate a pack against the target using probe system"""
        try:
            # Use probe evaluator for real emotion tracking
            # CRITICAL: Use the target model, not a test model
            result = self.probe_evaluator.evaluate_with_pack(
                pack_name=pack.name if hasattr(pack, 'name') else 'custom',
                test_prompts=test_prompts,
                model_name=self.model_name  # Use target model, not hardcoded test model
            )
            
            # Convert probe evaluation to target format
            actual_metrics = BehavioralMetrics()
            
            # Map probe emotions to target emotions
            for target_spec in target.targets:
                if target_spec.target_type.value == "emotion":
                    # Map probe emotions to target emotions
                    emotion_name = target_spec.name.replace('emotion_', '')
                    actual_value = result.emotions.get(emotion_name, 0.0)
                    actual_metrics.emotions[target_spec.name] = actual_value
                
                elif target_spec.target_type.value == "behavior":
                    # Map probe behaviors to target behaviors
                    behavior_name = target_spec.name.replace('behavior_', '')
                    # Use probe stats as behavior indicators
                    if behavior_name == 'creativity':
                        actual_value = result.probe_stats.get('NOVEL_LINK', {}).get('firing_rate', 0.0)
                    elif behavior_name == 'focus':
                        actual_value = result.probe_stats.get('FIXATION_FLOW', {}).get('firing_rate', 0.0)
                    elif behavior_name == 'socialization':
                        actual_value = result.latent_axes.get('sociality', 0.0)
                    else:
                        actual_value = 0.0
                    actual_metrics.behaviors[target_spec.name] = actual_value
                
                elif target_spec.target_type.value == "metric":
                    # Map probe metrics to target metrics
                    metric_name = target_spec.name.replace('metric_', '')
                    if metric_name == 'coherence':
                        actual_value = result.latent_axes.get('integration', 0.0)
                    elif metric_name == 'optimism':
                        actual_value = result.latent_axes.get('valence', 0.0)
                    elif metric_name == 'thoughtfulness':
                        actual_value = result.latent_axes.get('certainty', 0.0)
                    else:
                        actual_value = 0.0
                    actual_metrics.metrics[target_spec.name] = actual_value
            
            # Compute loss using target's loss function
            loss = target.compute_loss(actual_metrics.get_all_metrics())
            
            logger.debug(f"Pack evaluation: loss={loss:.4f}, emotions={result.emotions}")
            
            return loss
            
        except Exception as e:
            logger.error(f"Error evaluating pack with probe system: {e}")
            # Fallback to simple evaluation
            return self._evaluate_pack_simple(pack, target, test_prompts)
    
    def _evaluate_pack_simple(self, pack: Pack, target: BehavioralTarget, test_prompts: List[str]) -> float:
        """Fallback simple evaluation without probe system"""
        try:
            # Load model
            # CRITICAL: Use the target model, not a hardcoded test model
            # Transferability between architectures (GPT-2 vs Llama-3) is zero
            model, tokenizer, _ = self.model_manager.load_model(self.model_name)
            
            # Apply pack
            pack_manager = PackManager()
            pack_manager.apply_pack(pack, model)
            
            # Generate responses
            responses = []
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_new_tokens=50,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
            
            # Evaluate responses
            actual_metrics = self.evaluation_framework.evaluate_texts(responses)
            
            # Compute loss
            target_metrics = BehavioralMetrics()
            for target_spec in target.targets:
                if target_spec.target_type.value == "emotion":
                    target_metrics.emotions[target_spec.name] = target_spec.target_value
                elif target_spec.target_type.value == "behavior":
                    target_metrics.behaviors[target_spec.name] = target_spec.target_value
                elif target_spec.target_type.value == "metric":
                    target_metrics.metrics[target_spec.name] = target_spec.target_value
            
            loss = target.compute_loss(actual_metrics.get_all_metrics())
            
            return loss
            
        except Exception as e:
            logger.error(f"Error in simple pack evaluation: {e}")
            return float('inf')
    
    def _mutate(self, pack: Pack) -> Pack:
        """Mutate a pack using the evolutionary operators"""
        return self.mutator.mutate(pack)
    
    def _crossover(self, parent1: Pack, parent2: Pack) -> Tuple[Pack, Pack]:
        """Create offspring through crossover using evolutionary operators"""
        return self.crossover.crossover(parent1, parent2)
    
    def _tournament_selection(self, population: List[Pack], fitness_scores: List[float]) -> Pack:
        """Select individual using tournament selection"""
        tournament_size = min(3, len(population))
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmin(tournament_fitness)]
        return population[winner_idx]
    
    def _check_convergence(self, iteration_history: List[float]) -> bool:
        """Check if optimization has converged"""
        if len(iteration_history) < self.config.early_stopping_patience:
            return False
        
        recent_losses = iteration_history[-self.config.early_stopping_patience:]
        improvement = recent_losses[0] - recent_losses[-1]
        
        return improvement < self.config.convergence_threshold
    
    def _get_pack_parameter_bounds(self, pack: Pack) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization - includes all available effects"""
        # Get all available effects directly from the effect registry
        from ..effects import EffectRegistry
        effect_registry = EffectRegistry()
        available_effects = effect_registry.list_effects()
        
        bounds = []
        
        # For each available effect, add parameters:
        # 1. Effect selection (0 or 1)
        # 2. Effect weight (0-1) 
        # 3. Effect strength (0-1)
        # 4. Effect mode (0-1, maps to low/medium/high)
        for effect_name in available_effects:
            bounds.extend([
                (0.0, 1.0),  # Effect selection (binary)
                (0.0, 1.0),  # Effect weight
                (0.0, 1.0),  # Effect strength
                (0.0, 1.0),  # Effect mode
            ])
        
        return bounds
    
    def _create_pack_from_parameters(self, base_pack: Pack, params: np.ndarray) -> Pack:
        """Create a pack with given parameters - can include any available effects"""
        # Get all available effects directly from the effect registry
        from ..effects import EffectRegistry
        effect_registry = EffectRegistry()
        available_effects = effect_registry.list_effects()
        
        # Create a new pack with effects based on parameters
        from ..pack_system import EffectConfig
        new_pack = Pack(
            name=f"optimized_{base_pack.name}",
            description=f"Optimized version of {base_pack.name}",
            effects=[]
        )
        
        param_idx = 0
        
        # For each available effect, check if it should be included
        for effect_name in available_effects:
            if param_idx + 3 < len(params):  # Need at least 4 parameters per effect
                # Check if effect should be included (selection > 0.5)
                if params[param_idx] > 0.5:
                    # Get effect parameters
                    weight = np.clip(params[param_idx + 1], 0.0, 1.0)
                    strength = np.clip(params[param_idx + 2], 0.0, 1.0)
                    mode_val = np.clip(params[param_idx + 3], 0.0, 1.0)
                    
                    # Map mode value to string
                    if mode_val < 0.33:
                        mode = 'low'
                    elif mode_val < 0.66:
                        mode = 'medium'
                    else:
                        mode = 'high'
                    
                    # Create effect configuration
                    effect_config = EffectConfig(
                        effect=effect_name,
                        weight=weight,
                        direction='up',  # Default direction
                        parameters={
                            'strength': strength,
                            'mode': mode,
                            'enabled': True
                        }
                    )
                    
                    new_pack.effects.append(effect_config)
                
                param_idx += 4  # Move to next effect's parameters
        
        return new_pack

# Convenience functions
def optimize_pack_for_target(pack: Pack, 
                           target: BehavioralTarget,
                           model_manager: ModelSupportManager,
                           method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
                           model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> OptimizationResult:
    """
    Convenience function to optimize a pack for a target.
    
    Args:
        pack: Pack to optimize
        target: Behavioral target to optimize for
        model_manager: Model support manager
        method: Optimization method to use
        model_name: Target model for optimization (default: Llama-3.1-8B-Instruct)
                   CRITICAL: Must match the model used for final evaluation.
    
    Returns:
        OptimizationResult with optimized pack
    """
    evaluation_framework = EvaluationFramework()
    config = OptimizationConfig(method=method, model_name=model_name)
    optimizer = PackOptimizer(model_manager, evaluation_framework, config)
    
    return optimizer.optimize_pack(pack, target)

def create_optimized_pack(target_name: str, 
                         base_pack_name: str = "none",
                         method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
                         model_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> Pack:
    """
    Create an optimized pack for a target.
    
    Args:
        target_name: Name of the behavioral target
        base_pack_name: Name of the base pack to optimize
        method: Optimization method to use
        model_name: Target model for optimization (default: Llama-3.1-8B-Instruct)
                   CRITICAL: Must match the model used for final evaluation.
    
    Returns:
        Optimized pack
    """
    # Get target
    target_manager = TargetManager()
    target = target_manager.get_target(target_name)
    if not target:
        raise ValueError(f"Target not found: {target_name}")
    
    # Get base pack
    pack_registry = PackRegistry()
    base_pack = pack_registry.get_pack(base_pack_name)
    if not base_pack:
        raise ValueError(f"Base pack not found: {base_pack_name}")
    
    # Optimize
    # CRITICAL: Use production mode if using production model, test mode for test models
    test_mode = "test" in model_name.lower() or "gpt2" in model_name.lower() or "dialo" in model_name.lower()
    model_manager = ModelSupportManager(test_mode=test_mode)
    result = optimize_pack_for_target(base_pack, target, model_manager, method, model_name=model_name)
    
    return result.optimized_pack
