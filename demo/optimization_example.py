#!/usr/bin/env python3
"""
Pack Optimization Example

This script demonstrates how to use the neuromodulation pack optimization framework
to create custom behavioral effects through machine learning.

Example: Optimizing a pack to create an "alcohol-like" intoxicant effect.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuromod.optimization.targets import BehavioralTarget, TargetSpec, TargetType, OptimizationObjective
from neuromod.optimization.pack_optimizer import PackOptimizer, OptimizationConfig, OptimizationMethod
from neuromod.optimization.evaluation import EvaluationFramework
from neuromod.model_support import ModelSupportManager
from neuromod.pack_system import PackRegistry

def create_alcohol_target():
    """Create a behavioral target that mimics alcohol effects"""
    return BehavioralTarget(
        name="alcohol_intoxicant",
        description="Alcohol effects: increased socialization, joy, reduced coherence, lower anxiety",
        targets=[
            # Social and emotional effects
            TargetSpec("behavior_socialization", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.8, 1.0),
            TargetSpec("emotion_joy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.7, 1.0),
            TargetSpec("latent_sociality", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            
            # Cognitive effects (alcohol reduces coherence and focus)
            TargetSpec("metric_coherence", TargetType.METRIC, OptimizationObjective.MINIMIZE, 0.3, 0.8),
            TargetSpec("behavior_focus", TargetType.BEHAVIOR, OptimizationObjective.MINIMIZE, 0.2, 0.6),
            TargetSpec("latent_certainty", TargetType.LATENT_AXIS, OptimizationObjective.MINIMIZE, 0.3, 0.7),
            
            # Anxiety reduction
            TargetSpec("emotion_anxiety", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.2, 0.8),
            TargetSpec("emotion_fear", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.1, 0.6),
            
            # Slight increase in risk-taking
            TargetSpec("latent_risk_preference", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.6, 0.5)
        ]
    )

def create_mdma_target():
    """Create a behavioral target that mimics MDMA effects"""
    return BehavioralTarget(
        name="mdma_ecstasy",
        description="MDMA effects: extreme empathy, euphoria, social bonding, anxiety reduction",
        targets=[
            # Empathy & Social Bonding
            TargetSpec("emotion_empathy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            TargetSpec("behavior_socialization", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.95, 1.0),
            TargetSpec("latent_sociality", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            
            # Euphoria & Joy
            TargetSpec("emotion_joy", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.85, 1.0),
            TargetSpec("emotion_awe", TargetType.EMOTION, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            TargetSpec("latent_valence", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.9, 1.0),
            
            # Anxiety & Fear Reduction
            TargetSpec("emotion_anxiety", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.05, 1.0),
            TargetSpec("emotion_fear", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.05, 0.8),
            TargetSpec("emotion_anger", TargetType.EMOTION, OptimizationObjective.MINIMIZE, 0.1, 0.6),
            
            # Sensory Enhancement
            TargetSpec("latent_arousal", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            TargetSpec("behavior_creativity", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.7, 0.7),
            TargetSpec("metric_originality", TargetType.METRIC, OptimizationObjective.MAXIMIZE, 0.8, 0.6),
            
            # Emotional Openness
            TargetSpec("latent_openness", TargetType.LATENT_AXIS, OptimizationObjective.MAXIMIZE, 0.8, 0.8),
            TargetSpec("behavior_reflection", TargetType.BEHAVIOR, OptimizationObjective.MAXIMIZE, 0.7, 0.6)
        ]
    )

def run_optimization_example():
    """Run a complete optimization example"""
    print("🧠 Neuromodulation Pack Optimization Example")
    print("=" * 50)
    
    # Initialize components
    print("📋 Initializing components...")
    model_manager = ModelSupportManager(test_mode=True)
    evaluation_framework = EvaluationFramework()
    pack_registry = PackRegistry()
    
    # Get base pack (we'll use a simple pack as starting point)
    base_pack = pack_registry.get_pack("placebo")  # Start with placebo as baseline
    print(f"📦 Base pack: {base_pack.name} ({len(base_pack.effects)} effects)")
    
    # Create target
    target = create_alcohol_target()
    print(f"🎯 Target: {target.name}")
    print(f"   Description: {target.description}")
    print(f"   Number of targets: {len(target.targets)}")
    
    # Test prompts
    test_prompts = [
        "Tell me about your feelings",
        "What makes you happy?",
        "Describe a social situation you enjoyed",
        "How do you handle stress?",
        "What's your favorite way to relax?"
    ]
    print(f"💬 Test prompts: {len(test_prompts)}")
    
    # Configure optimization
    config = OptimizationConfig(
        method=OptimizationMethod.EVOLUTIONARY,
        max_iterations=10,  # Quick demo
        population_size=8,
        mutation_rate=0.4,
        crossover_rate=0.7
    )
    print(f"⚙️  Optimization method: {config.method.value}")
    print(f"   Max iterations: {config.max_iterations}")
    print(f"   Population size: {config.population_size}")
    
    # Create optimizer
    optimizer = PackOptimizer(model_manager, evaluation_framework, config)
    
    print("\n🚀 Starting optimization...")
    print("   This may take a few minutes...")
    
    # Run optimization
    result = optimizer.optimize_pack(base_pack, target, test_prompts)
    
    print("\n✅ Optimization Complete!")
    print("=" * 30)
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Iterations: {result.iterations}")
    print(f"Converged: {result.converged}")
    print(f"Success: {result.success}")
    
    print(f"\n📦 Optimized Pack: {result.optimized_pack.name}")
    print(f"   Description: {result.optimized_pack.description}")
    print(f"   Effects: {len(result.optimized_pack.effects)}")
    
    print("\n🔧 Optimized Effects:")
    for i, effect in enumerate(result.optimized_pack.effects, 1):
        print(f"   {i:2d}. {effect.effect}")
        print(f"       Weight: {effect.weight:.3f}, Direction: {effect.direction}")
        if effect.parameters:
            print(f"       Parameters: {effect.parameters}")
        print()
    
    # Show improvement
    print("📊 Optimization Results:")
    print(f"   Target loss: {result.final_loss:.4f}")
    print(f"   Iterations run: {result.iterations}")
    print(f"   Convergence: {'Yes' if result.converged else 'No'}")
    print(f"   Success: {'Yes' if result.success else 'No'}")
    
    if result.iteration_history:
        print(f"   Loss progression: {[f'{loss:.3f}' for loss in result.iteration_history[:5]]}...")
    
    print("\n🎉 Example completed successfully!")
    print("   The optimization framework successfully created a custom pack")
    print("   optimized for alcohol-like behavioral effects.")
    
    return result

def main():
    """Main function"""
    try:
        result = run_optimization_example()
        
        print("\n" + "=" * 50)
        print("📚 Next Steps:")
        print("   1. Try different optimization methods (Bayesian, RL)")
        print("   2. Create custom behavioral targets")
        print("   3. Use the CLI for interactive optimization")
        print("   4. Explore the full documentation in neuromod/optimization/")
        print("\n🔗 Useful Commands:")
        print("   python -m neuromod.optimization.cli test-pack --target alcohol_intoxicant")
        print("   python -m neuromod.optimization.cli optimize --pack placebo --target alcohol_intoxicant")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("   Make sure you're in the project root directory")
        print("   and have installed all dependencies.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
