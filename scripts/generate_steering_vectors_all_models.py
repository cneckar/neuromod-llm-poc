#!/usr/bin/env python3
"""
Generate steering vectors for ALL available models.

This script iterates through all available models and generates steering vectors
for each one using the same pipeline as generate_steering_vectors.py.

Usage:
    # Generate for all production models
    python scripts/generate_steering_vectors_all_models.py
    
    # Generate for test models only
    python scripts/generate_steering_vectors_all_models.py --test-mode
    
    # Generate for specific models only
    python scripts/generate_steering_vectors_all_models.py --models gpt2 meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys
import subprocess
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neuromod.model_support import ModelSupportManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate steering vectors for all available models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-mode", action="store_true",
                       help="Generate for test mode models only")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to generate vectors for (default: all available)")
    parser.add_argument("--output-dir", type=str, default="outputs/steering_vectors",
                       help="Output directory for steering vectors")
    parser.add_argument("--steering-type", type=str, default=None,
                       help="Generate only this steering type (default: all types)")
    parser.add_argument("--layer", type=int, default=None,
                       help="Specific layer to use (default: None = use all layers)")
    parser.add_argument("--no-pca", action="store_true",
                       help="Disable PCA denoising")
    parser.add_argument("--no-validate", action="store_true",
                       help="Skip validation")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip models that already have steering vectors")
    parser.add_argument("--min-pairs", type=int, default=100,
                       help="Minimum number of prompt pairs required (default: 100, lower values may reduce quality)")
    
    args = parser.parse_args()
    
    # Get available models
    model_manager = ModelSupportManager(test_mode=args.test_mode)
    if args.models:
        available_models = args.models
        logger.info(f"Using specified models: {available_models}")
    else:
        available_models = model_manager.get_available_models()
        logger.info(f"Found {len(available_models)} available model(s)")
    
    # Filter out mock models
    available_models = [m for m in available_models if m != "mock"]
    
    if not available_models:
        logger.error("No models available to process")
        return 1
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating steering vectors for {len(available_models)} model(s)")
    logger.info(f"{'='*60}\n")
    
    # Build base command
    script_path = Path(__file__).parent / "generate_steering_vectors.py"
    
    # Create base output directory
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track results
    results = {
        "success": [],
        "failed": [],
        "skipped": []
    }
    
    # Process each model
    for model_name in available_models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing model: {model_name}")
        logger.info(f"{'='*60}")
        
        # Create model-specific output directory
        # Sanitize model name for filesystem (replace / with _)
        model_safe_name = model_name.replace("/", "_").replace("\\", "_")
        model_output_dir = base_output_dir / model_safe_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if vectors already exist (if skip-existing is enabled)
        if args.skip_existing:
            # Check if any steering vectors exist for this model
            if any(model_output_dir.glob("*.pt")):
                logger.info(f"⏭️  Skipping {model_name} (vectors already exist in {model_output_dir})")
                results["skipped"].append(model_name)
                continue
        
        # Build command for this model
        base_cmd = [
            sys.executable,
            str(script_path),
            "--model", model_name,
            "--output-dir", str(model_output_dir)
        ]
        
        if args.steering_type:
            base_cmd.extend(["--steering-type", args.steering_type])
        if args.layer is not None:
            base_cmd.extend(["--layer", str(args.layer)])
        if args.no_pca:
            base_cmd.append("--no-pca")
        if args.no_validate:
            base_cmd.append("--no-validate")
        if args.test_mode:
            base_cmd.append("--test-mode")
        if args.min_pairs != 100:  # Only add if different from default
            base_cmd.extend(["--min-pairs", str(args.min_pairs)])
        
        try:
            # Run the generation script
            result = subprocess.run(
                base_cmd,
                capture_output=False,  # Show output in real-time
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Successfully generated vectors for {model_name}")
                results["success"].append(model_name)
            else:
                logger.error(f"❌ Failed to generate vectors for {model_name}")
                results["failed"].append(model_name)
                
        except Exception as e:
            logger.error(f"❌ Error processing {model_name}: {e}")
            results["failed"].append(model_name)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"GENERATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Successful: {len(results['success'])}/{len(available_models)}")
    if results["success"]:
        logger.info(f"   Models: {', '.join(results['success'])}")
    
    if results["skipped"]:
        logger.info(f"⏭️  Skipped: {len(results['skipped'])}")
        logger.info(f"   Models: {', '.join(results['skipped'])}")
    
    if results["failed"]:
        logger.warning(f"❌ Failed: {len(results['failed'])}")
        logger.warning(f"   Models: {', '.join(results['failed'])}")
    
    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

