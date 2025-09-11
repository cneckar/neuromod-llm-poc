"""
Command Line Interface for Drug Design Laboratory

Provides easy-to-use CLI commands for designing and optimizing neuromodulation packs.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .laboratory import DrugDesignLab, create_lab
from .targets import TargetManager, BehavioralTarget
from .pack_optimizer import OptimizationMethod
from .evaluation import EvaluationFramework

logger = logging.getLogger(__name__)

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cmd_list_targets(args):
    """List available behavioral targets"""
    target_manager = TargetManager()
    targets = target_manager.list_targets()
    
    print("Available Behavioral Targets:")
    print("=" * 40)
    
    for target_name in targets:
        target = target_manager.get_target(target_name)
        print(f"\n{target_name}:")
        print(f"  Description: {target.description}")
        print(f"  Targets: {[t.name for t in target.targets]}")
        print(f"  Test prompts: {len(target.test_prompts)}")

def cmd_create_target(args):
    """Create a new behavioral target"""
    target_manager = TargetManager()
    
    # Create target
    target = target_manager.create_target(args.name, args.description)
    
    # Add targets from command line
    if args.emotions:
        for emotion, value in args.emotions.items():
            target.add_emotion_target(emotion, float(value))
    
    if args.behaviors:
        for behavior, value in args.behaviors.items():
            target.add_behavior_target(behavior, float(value))
    
    if args.metrics:
        for metric, value in args.metrics.items():
            target.add_metric_target(metric, float(value))
    
    # Add test prompts
    if args.prompts:
        target.test_prompts = args.prompts
    
    # Save if requested
    if args.save:
        filepath = Path(args.save)
        target_manager.save_target(target, str(filepath))
        print(f"Target saved to: {filepath}")
    
    print(f"Created target: {target.name}")
    print(f"Description: {target.description}")
    print(f"Targets: {[t.name for t in target.targets]}")

def cmd_design_drug(args):
    """Design a drug for a specific target"""
    lab = create_lab()
    
    print(f"Creating laboratory session for target: {args.target}")
    
    # Create session
    session = lab.create_session(args.target, args.base_pack)
    
    print(f"Session created: {session.session_id}")
    print(f"Base pack: {args.base_pack}")
    
    # Optimize pack
    method = OptimizationMethod(args.method)
    print(f"Starting optimization with method: {method.value}")
    
    result = lab.optimize_pack(
        session.session_id,
        method=method,
        max_iterations=args.iterations
    )
    
    print(f"Optimization completed!")
    print(f"Final loss: {result.final_loss:.4f}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.convergence_iteration}")
    
    # Test the optimized pack
    if args.test:
        print("\nTesting optimized pack...")
        test_result = lab.test_pack(session.session_id)
        print(f"Test loss: {test_result['target_loss']:.4f}")
    
    # Save session if requested
    if args.save:
        filepath = Path(args.save)
        lab.save_session(session.session_id, str(filepath))
        print(f"Session saved to: {filepath}")

def cmd_test_pack(args):
    """Test a pack with custom prompts"""
    lab = create_lab()
    
    # Load session if provided
    if args.session_file:
        session_id = lab.load_session(args.session_file)
        print(f"Loaded session: {session_id}")
    else:
        # Create new session
        session = lab.create_session(args.target, args.base_pack)
        session_id = session.session_id
        print(f"Created session: {session_id}")
    
    # Test prompts
    if args.prompts:
        test_prompts = args.prompts
    else:
        # Use default prompts from target
        target = lab.target_manager.get_target(args.target)
        test_prompts = target.test_prompts
    
    print(f"Testing with {len(test_prompts)} prompts...")
    
    # Test pack
    test_result = lab.test_pack(session_id, test_prompts)
    
    print(f"\nTest Results:")
    print(f"Target loss: {test_result['target_loss']:.4f}")
    print(f"Metrics: {json.dumps(test_result['metrics'], indent=2)}")
    
    # Show responses
    if args.show_responses:
        print(f"\nResponses:")
        for i, (prompt, response) in enumerate(zip(test_prompts, test_result['responses'])):
            print(f"\n{i+1}. Prompt: {prompt}")
            print(f"   Response: {response}")

def cmd_compare_packs(args):
    """Compare base pack vs optimized pack"""
    lab = create_lab()
    
    # Load session
    if args.session_file:
        session_id = lab.load_session(args.session_file)
    else:
        raise ValueError("Session file required for comparison")
    
    print(f"Comparing packs for session: {session_id}")
    
    # Compare packs
    comparison = lab.compare_packs(session_id)
    
    print(f"\nComparison Results:")
    print(f"Base pack loss: {comparison['base_pack']['target_loss']:.4f}")
    print(f"Optimized pack loss: {comparison['optimized_pack']['target_loss']:.4f}")
    print(f"Loss reduction: {comparison['improvement']['loss_reduction']:.4f}")
    print(f"Relative improvement: {comparison['improvement']['relative_improvement']:.2%}")
    
    # Show detailed metrics
    if args.detailed:
        print(f"\nDetailed Metrics:")
        print(f"Base pack metrics: {json.dumps(comparison['base_pack']['metrics'], indent=2)}")
        print(f"Optimized pack metrics: {json.dumps(comparison['optimized_pack']['metrics'], indent=2)}")

def cmd_list_sessions(args):
    """List all laboratory sessions"""
    lab = create_lab()
    sessions = lab.list_sessions()
    
    print("Laboratory Sessions:")
    print("=" * 50)
    
    for session in sessions:
        print(f"\nSession: {session['session_id']}")
        print(f"  Target: {session['target']['name']}")
        print(f"  Base pack: {session['base_pack']}")
        print(f"  Optimized: {session['optimized_pack'] or 'No'}")
        print(f"  Final loss: {session['optimization_result']['final_loss'] or 'N/A'}")
        print(f"  Test results: {session['test_results_count']}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Drug Design Laboratory CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available targets
  python -m neuromod.optimization.cli list-targets
  
  # Design a joyful social drug
  python -m neuromod.optimization.cli design-drug --target joyful_social --method random_search
  
  # Test a pack with custom prompts
  python -m neuromod.optimization.cli test-pack --target joyful_social --prompts "Tell me about happiness" "What makes you smile?"
  
  # Compare base vs optimized pack
  python -m neuromod.optimization.cli compare-packs --session-file session.json
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List targets command
    list_targets_parser = subparsers.add_parser('list-targets', help='List available targets')
    list_targets_parser.set_defaults(func=cmd_list_targets)
    
    # Create target command
    create_target_parser = subparsers.add_parser('create-target', help='Create a new target')
    create_target_parser.add_argument('--name', required=True, help='Target name')
    create_target_parser.add_argument('--description', required=True, help='Target description')
    create_target_parser.add_argument('--emotions', type=json.loads, help='Emotion targets (JSON)')
    create_target_parser.add_argument('--behaviors', type=json.loads, help='Behavior targets (JSON)')
    create_target_parser.add_argument('--metrics', type=json.loads, help='Metric targets (JSON)')
    create_target_parser.add_argument('--prompts', nargs='+', help='Test prompts')
    create_target_parser.add_argument('--save', help='Save target to file')
    create_target_parser.set_defaults(func=cmd_create_target)
    
    # Design drug command
    design_drug_parser = subparsers.add_parser('design-drug', help='Design a drug for a target')
    design_drug_parser.add_argument('--target', required=True, help='Target name')
    design_drug_parser.add_argument('--base-pack', default='none', help='Base pack name')
    design_drug_parser.add_argument('--method', default='random_search', 
                                   choices=['random_search', 'evolutionary', 'gradient_descent'],
                                   help='Optimization method')
    design_drug_parser.add_argument('--iterations', type=int, default=50, help='Max iterations')
    design_drug_parser.add_argument('--test', action='store_true', help='Test optimized pack')
    design_drug_parser.add_argument('--save', help='Save session to file')
    design_drug_parser.set_defaults(func=cmd_design_drug)
    
    # Test pack command
    test_pack_parser = subparsers.add_parser('test-pack', help='Test a pack')
    test_pack_parser.add_argument('--target', help='Target name')
    test_pack_parser.add_argument('--base-pack', default='none', help='Base pack name')
    test_pack_parser.add_argument('--session-file', help='Load existing session')
    test_pack_parser.add_argument('--prompts', nargs='+', help='Test prompts')
    test_pack_parser.add_argument('--show-responses', action='store_true', help='Show model responses')
    test_pack_parser.set_defaults(func=cmd_test_pack)
    
    # Compare packs command
    compare_packs_parser = subparsers.add_parser('compare-packs', help='Compare base vs optimized pack')
    compare_packs_parser.add_argument('--session-file', required=True, help='Session file')
    compare_packs_parser.add_argument('--detailed', action='store_true', help='Show detailed metrics')
    compare_packs_parser.set_defaults(func=cmd_compare_packs)
    
    # List sessions command
    list_sessions_parser = subparsers.add_parser('list-sessions', help='List all sessions')
    list_sessions_parser.set_defaults(func=cmd_list_sessions)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Run command
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
