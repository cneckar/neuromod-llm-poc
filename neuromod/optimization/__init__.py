"""
Pack Optimization System

This module provides tools for optimizing neuromodulation packs to achieve
specific behavioral and emotional outcomes through iterative refinement.

Key Components:
- TargetDefinition: Define behavioral/emotional targets
- PackOptimizer: Optimize pack parameters to meet targets
- EvaluationFramework: Measure how well packs achieve targets
- DrugDesignLab: Interactive laboratory interface
"""

from .targets import (
    BehavioralTarget, TargetManager, 
    create_joyful_social_target, create_creative_focused_target, create_calm_reflective_target
)
from .pack_optimizer import PackOptimizer, OptimizationMethod
from .evaluation import EvaluationFramework, BehavioralMetrics
from .probe_evaluator import ProbeEvaluator, ProbeEvaluationResult
from .bayesian_optimizer import BayesianOptimizer, BayesianOptimizationConfig
from .rl_optimizer import RLOptimizer, RLOptimizationConfig
from .evolutionary_ops import PackMutator, PackCrossover, MutationConfig, CrossoverConfig
from .laboratory import DrugDesignLab, create_lab

__all__ = [
    'BehavioralTarget',
    'TargetManager', 
    'PackOptimizer',
    'OptimizationMethod',
    'EvaluationFramework',
    'BehavioralMetrics',
    'ProbeEvaluator',
    'ProbeEvaluationResult',
    'BayesianOptimizer',
    'BayesianOptimizationConfig',
    'RLOptimizer',
    'RLOptimizationConfig',
    'PackMutator',
    'PackCrossover',
    'MutationConfig',
    'CrossoverConfig',
    'DrugDesignLab',
    'create_lab',
    'create_joyful_social_target',
    'create_creative_focused_target', 
    'create_calm_reflective_target'
]
