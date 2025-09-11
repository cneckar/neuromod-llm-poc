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

@dataclass
class OptimizationResult:
    """Result of pack optimization"""
    optimized_pack: Pack
    final_loss: float
    iteration_history: List[float]
    best_loss: float
    convergence_iteration: int
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

class PackOptimizer:
    """Main pack optimization engine"""
    
    def __init__(self, 
                 model_manager: ModelSupportManager,
                 evaluation_framework: EvaluationFramework,
                 config: OptimizationConfig = None):
        self.model_manager = model_manager
        self.evaluation_framework = evaluation_framework
        self.config = config or OptimizationConfig()
        self.pack_registry = PackRegistry()
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
    
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
            metadata={'method': 'random_search'}
        )
    
    def _optimize_bayesian(self, 
                          pack: Pack, 
                          target: BehavioralTarget,
                          test_prompts: List[str]) -> OptimizationResult:
        """Optimize using Bayesian optimization"""
        
        # For now, fall back to random search
        # In a full implementation, we'd use a library like scikit-optimize
        logger.warning("Bayesian optimization not fully implemented - falling back to random search")
        return self._optimize_random_search(pack, target, test_prompts)
    
    def _optimize_reinforcement_learning(self, 
                                       pack: Pack, 
                                       target: BehavioralTarget,
                                       test_prompts: List[str]) -> OptimizationResult:
        """Optimize using reinforcement learning"""
        
        # For now, fall back to random search
        # In a full implementation, we'd use RL algorithms
        logger.warning("Reinforcement learning optimization not fully implemented - falling back to random search")
        return self._optimize_random_search(pack, target, test_prompts)
    
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
        """Evaluate a pack against the target"""
        try:
            # Load model
            model_name = "microsoft/DialoGPT-small"  # Use small model for speed
            model, tokenizer, _ = self.model_manager.load_model(model_name)
            
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
            logger.error(f"Error evaluating pack: {e}")
            return float('inf')
    
    def _mutate(self, pack: Pack) -> Pack:
        """Mutate a pack by randomly adjusting parameters"""
        # This is a simplified mutation - in practice, we'd need to know
        # which parameters are optimizable and how to adjust them
        
        # For now, just return a copy (no actual mutation)
        # In a full implementation, we'd adjust effect parameters
        return copy.deepcopy(pack)
    
    def _crossover(self, parent1: Pack, parent2: Pack) -> Tuple[Pack, Pack]:
        """Create offspring through crossover"""
        # This is a simplified crossover - in practice, we'd need to know
        # how to combine pack parameters
        
        # For now, just return copies of parents
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
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

# Convenience functions
def optimize_pack_for_target(pack: Pack, 
                           target: BehavioralTarget,
                           model_manager: ModelSupportManager,
                           method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH) -> OptimizationResult:
    """Convenience function to optimize a pack for a target"""
    
    evaluation_framework = EvaluationFramework()
    config = OptimizationConfig(method=method)
    optimizer = PackOptimizer(model_manager, evaluation_framework, config)
    
    return optimizer.optimize_pack(pack, target)

def create_optimized_pack(target_name: str, 
                         base_pack_name: str = "none",
                         method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH) -> Pack:
    """Create an optimized pack for a target"""
    
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
    model_manager = ModelSupportManager(test_mode=True)
    result = optimize_pack_for_target(base_pack, target, model_manager, method)
    
    return result.optimized_pack
