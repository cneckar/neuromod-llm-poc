#!/usr/bin/env python3
"""
Generate steering vectors using Robust Mean Difference Vector (MDV) with PCA.

This script generates steering vectors using the robust pipeline:
1. Loads 100+ prompt pairs from datasets/steering_prompts.jsonl
2. Extracts activations from all layers
3. Applies PCA to difference vectors to extract PC1
4. Validates separation significance (p < 0.01)

Usage:
    python scripts/generate_steering_vectors.py --model <model_name> [--output-dir <dir>] [--steering-type <type>]
"""

import argparse
import sys
from pathlib import Path
import torch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.steering_generator import SteeringVectorGenerator
from neuromod.model_support import ModelSupportManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate robust steering vectors using MDV pipeline with PCA")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'gpt2', 'meta-llama/Llama-3.1-8B-Instruct')")
    parser.add_argument("--output-dir", type=str, default="outputs/steering_vectors",
                       help="Output directory for steering vectors (default: outputs/steering_vectors)")
    parser.add_argument("--dataset", type=str, default="datasets/steering_prompts.jsonl",
                       help="Path to JSONL dataset file (default: datasets/steering_prompts.jsonl)")
    parser.add_argument("--steering-type", type=str, default=None,
                       help="Generate only this steering type (default: all types in dataset)")
    parser.add_argument("--layer", type=int, default=None,
                       help="Specific layer to use (default: None = use all layers)")
    parser.add_argument("--no-pca", action="store_true",
                       help="Disable PCA denoising (use simple mean difference)")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation (not recommended)")
    parser.add_argument("--test-mode", action="store_true",
                       help="Use test mode (smaller models)")
    parser.add_argument("--min-pairs", type=int, default=100,
                       help="Minimum number of prompt pairs required (default: 100, lower values may reduce quality)")
    
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
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return 1
    
    # Get available steering types from dataset
    available_types = set()
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                import json
                data = json.loads(line.strip())
                if 'steering_type' in data:
                    available_types.add(data['steering_type'])
            except:
                continue
    
    if args.steering_type:
        if args.steering_type not in available_types:
            logger.error(f"Unknown steering type: {args.steering_type}")
            logger.info(f"Available types: {', '.join(sorted(available_types))}")
            return 1
        steering_types = [args.steering_type]
    else:
        steering_types = sorted(available_types)
    
    # Generate vectors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {len(steering_types)} steering vector(s) using robust MDV pipeline")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"PCA: {'disabled' if args.no_pca else 'enabled'}")
    logger.info(f"Validation: {'disabled' if args.no_validate else 'enabled'}")
    
    success_count = 0
    failed_types = []
    
    for steering_type in steering_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating vector for: {steering_type}")
        logger.info(f"{'='*60}")
        
        try:
            # Use robust method
            steering_vec = generator.compute_vector_robust(
                dataset_path=dataset_path,
                steering_type=steering_type,
                layer_idx=args.layer,
                use_pca=not args.no_pca,
                validate=not args.no_validate,
                min_pairs=args.min_pairs
            )
            
            # Save vector
            layer_suffix = f"_layer{args.layer}" if args.layer is not None else "_all_layers"
            output_path = output_dir / f"{steering_type}{layer_suffix}.pt"
            torch.save(steering_vec, output_path)
            
            logger.info(f"✓ Successfully generated: {output_path}")
            logger.info(f"  Vector shape: {steering_vec.shape}, norm: {torch.norm(steering_vec):.4f}")
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

