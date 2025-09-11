#!/usr/bin/env python3
"""
Pack Optimization Demo

Demonstrates the drug design laboratory functionality for optimizing
neuromodulation packs to achieve specific behavioral targets.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from neuromod.optimization import (
    DrugDesignLab, create_joyful_social_target,
    create_creative_focused_target, create_calm_reflective_target
)
from neuromod.optimization.laboratory import create_lab
from neuromod.optimization.pack_optimizer import OptimizationMethod

def demo_target_creation():
    """Demonstrate creating custom behavioral targets"""
    print("🎯 DEMO: Creating Custom Behavioral Targets")
    print("=" * 50)
    
    from neuromod.optimization.targets import TargetManager
    
    target_manager = TargetManager()
    
    # Create a custom target
    target = target_manager.create_target(
        name="motivational_energetic",
        description="Increase motivation, energy, and productivity while maintaining focus"
    )
    
    # Add specific targets
    target.add_emotion_target("joy", 0.7, weight=1.5)
    target.add_emotion_target("excitement", 0.6, weight=1.0)
    target.add_behavior_target("productivity", 0.8, weight=2.0)
    target.add_behavior_target("focus", 0.7, weight=1.5)
    target.add_metric_target("optimism", 0.6, weight=1.0)
    
    # Add test prompts
    target.test_prompts = [
        "What are your goals for today?",
        "How do you stay motivated?",
        "Describe your ideal work environment",
        "What energizes you most?",
        "How do you maintain focus?"
    ]
    
    print(f"Created target: {target.name}")
    print(f"Description: {target.description}")
    print(f"Targets: {[t.name for t in target.targets]}")
    print(f"Test prompts: {len(target.test_prompts)}")
    
    return target

def demo_evaluation_framework():
    """Demonstrate the evaluation framework"""
    print("\n🔍 DEMO: Evaluation Framework")
    print("=" * 50)
    
    from neuromod.optimization.evaluation import EvaluationFramework
    
    evaluator = EvaluationFramework()
    
    # Test texts
    test_texts = [
        "I am so happy and excited about this wonderful opportunity!",
        "This is a sad and depressing situation that makes me feel hopeless.",
        "I feel calm and peaceful, ready to reflect on my experiences."
    ]
    
    print("Analyzing texts:")
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. {text}")
        
        metrics = evaluator.evaluate_text(text)
        print(f"   Emotions: {metrics.emotions}")
        print(f"   Behaviors: {metrics.behaviors}")
        print(f"   Metrics: {metrics.metrics}")
    
    # Analyze all texts together
    print(f"\nOverall analysis:")
    overall_metrics = evaluator.evaluate_texts(test_texts)
    print(f"Average emotions: {overall_metrics.emotions}")
    print(f"Average behaviors: {overall_metrics.behaviors}")
    print(f"Average metrics: {overall_metrics.metrics}")

def demo_drug_design_lab():
    """Demonstrate the drug design laboratory"""
    print("\n🧪 DEMO: Drug Design Laboratory")
    print("=" * 50)
    
    # Create laboratory
    lab = create_lab()
    
    # List available targets
    print("Available targets:")
    targets = lab.target_manager.list_targets()
    for target_name in targets:
        print(f"  - {target_name}")
    
    # Create a session for joyful social target
    print(f"\nCreating session for 'joyful_social' target...")
    session = lab.create_session("joyful_social", "none")
    print(f"Session created: {session.session_id}")
    print(f"Target: {session.target.name}")
    print(f"Base pack: {session.base_pack.name if hasattr(session.base_pack, 'name') else 'unknown'}")
    
    # Test the base pack
    print(f"\nTesting base pack...")
    try:
        test_result = lab.test_pack(session.session_id)
        print(f"Base pack test loss: {test_result['target_loss']:.4f}")
        print(f"Metrics: {test_result['metrics']}")
    except Exception as e:
        print(f"Error testing base pack: {e}")
        print("This is expected if model loading fails in demo mode")
    
    # Show session summary
    print(f"\nSession summary:")
    summary = lab.get_session_summary(session.session_id)
    print(f"  Target: {summary['target']['name']}")
    print(f"  Test results: {summary['test_results_count']}")
    print(f"  Duration: {summary['duration']:.1f} seconds")

def demo_optimization_methods():
    """Demonstrate different optimization methods"""
    print("\n⚙️ DEMO: Optimization Methods")
    print("=" * 50)
    
    from neuromod.optimization.pack_optimizer import OptimizationMethod
    
    methods = [
        OptimizationMethod.RANDOM_SEARCH,
        OptimizationMethod.EVOLUTIONARY,
        OptimizationMethod.GRADIENT_DESCENT,
        OptimizationMethod.BAYESIAN,
        OptimizationMethod.REINFORCEMENT_LEARNING
    ]
    
    print("Available optimization methods:")
    for method in methods:
        print(f"  - {method.value}")
    
    print(f"\nNote: Some methods are not fully implemented yet and fall back to random search")

def demo_preset_targets():
    """Demonstrate preset behavioral targets"""
    print("\n📋 DEMO: Preset Behavioral Targets")
    print("=" * 50)
    
    # Get preset targets
    joyful_social = create_joyful_social_target()
    creative_focused = create_creative_focused_target()
    calm_reflective = create_calm_reflective_target()
    
    targets = [joyful_social, creative_focused, calm_reflective]
    
    for target in targets:
        print(f"\n{target.name.upper()}:")
        print(f"  Description: {target.description}")
        print(f"  Targets: {[t.name for t in target.targets]}")
        print(f"  Test prompts: {len(target.test_prompts)}")
        
        # Show sample prompts
        if target.test_prompts:
            print(f"  Sample prompt: '{target.test_prompts[0]}'")

def main():
    """Run the pack optimization demo"""
    print("🧪 PACK OPTIMIZATION DEMO")
    print("=" * 60)
    print("Demonstrating the drug design laboratory for neuromodulation packs")
    print("=" * 60)
    
    try:
        # Run demos
        demo_preset_targets()
        demo_target_creation()
        demo_evaluation_framework()
        demo_optimization_methods()
        demo_drug_design_lab()
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        print("\nNext steps:")
        print("1. Run: python -m neuromod.optimization.cli list-targets")
        print("2. Run: python -m neuromod.optimization.cli design-drug --target joyful_social")
        print("3. Run: python -m neuromod.optimization.cli test-pack --target joyful_social")
        
    except Exception as e:
        print(f"\n❌ DEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
