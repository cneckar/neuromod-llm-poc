#!/usr/bin/env python3
"""
Generate steering vectors using Contrastive Activation Addition (CAA).

This script generates steering vectors for all defined steering types by
computing the difference between activations from positive and negative prompts.

Usage:
    python scripts/generate_steering_vectors.py --model <model_name> [--layer <layer_idx>] [--output-dir <dir>]
"""

import argparse
import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.steering_generator import SteeringVectorGenerator, CONTRASTIVE_PAIRS
from neuromod.model_support import ModelSupportManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate steering vectors using CAA")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt2', 'meta-llama/Llama-3.1-8B-Instruct')")
    parser.add_argument("--layer", type=int, default=-1,
                       help="Layer index to extract from (-1 for last layer, default: -1)")
    parser.add_argument("--output-dir", type=str, default="outputs/steering_vectors",
                       help="Output directory for steering vectors (default: outputs/steering_vectors)")
    parser.add_argument("--steering-type", type=str, default=None,
                       help="Generate only this steering type (default: all types)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use test mode (smaller models)")
    
    args = parser.parse_args()
    
    # Initialize model support
    model_manager = ModelSupportManager(test_mode=args.test_mode)
    
    logger.info(f"Loading model: {args.model}")
    try:
        model, tokenizer, model_info = model_manager.load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to load model {args.model}: {e}")
        return 1
    
    # Initialize generator
    generator = SteeringVectorGenerator(model, tokenizer)
    
    # Determine which steering types to generate
    if args.steering_type:
        if args.steering_type not in CONTRASTIVE_PAIRS:
            logger.error(f"Unknown steering type: {args.steering_type}")
            logger.info(f"Available types: {', '.join(CONTRASTIVE_PAIRS.keys())}")
            return 1
        steering_types = [args.steering_type]
    else:
        steering_types = list(CONTRASTIVE_PAIRS.keys())
    
    # Generate vectors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {len(steering_types)} steering vector(s) for layer {args.layer}")
    logger.info(f"Output directory: {output_dir}")
    
    success_count = 0
    failed_types = []
    
    for steering_type in steering_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating vector for: {steering_type}")
        logger.info(f"{'='*60}")
        
        try:
            pairs = CONTRASTIVE_PAIRS[steering_type]
            positive_prompts = pairs["positive"]
            negative_prompts = pairs["negative"]
            
            # Generate and save
            output_path = generator.generate_and_save(
                steering_type=steering_type,
                positive_prompts=positive_prompts,
                negative_prompts=negative_prompts,
                layer_idx=args.layer,
                output_dir=output_dir
            )
            
            logger.info(f"✓ Successfully generated: {output_path}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"✗ Failed to generate vector for {steering_type}: {e}")
            failed_types.append(steering_type)
            import traceback
            traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully generated: {success_count}/{len(steering_types)}")
    if failed_types:
        logger.warning(f"Failed types: {', '.join(failed_types)}")
    
    # Cleanup
    model_manager.unload_model(args.model)
    
    return 0 if len(failed_types) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

