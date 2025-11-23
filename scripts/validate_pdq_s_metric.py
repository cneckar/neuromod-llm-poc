#!/usr/bin/env python3
"""
Validate PDQ-S Metric Using Permutation Test

This script validates that the PDQ-S (Psychedelic Detection Questionnaire)
metric is actually detecting psychedelic effects, not just random noise or
confounds like sentence length.

Usage:
    python scripts/validate_pdq_s_metric.py --lsd-results <file> --placebo-results <file> [--output <file>]
"""

import argparse
import json
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.permutation_test import validate_pdq_s_metric, PermutationTestValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_results_from_file(file_path: Path) -> list:
    """Load PDQ-S results from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different file formats
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # Try to extract results
        if 'results' in data:
            return data['results']
        elif 'pdq_s_results' in data:
            return data['pdq_s_results']
        elif 'test_results' in data:
            return data['test_results']
        else:
            # Assume the dict itself contains the results
            return [data]
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def extract_pdq_s_scores(results: list) -> list:
    """Extract PDQ-S scores from results"""
    extracted = []
    
    for result in results:
        # Try different possible field names
        score_dict = {}
        
        if isinstance(result, dict):
            # Look for PDQ-S specific fields
            if 'presence_probability' in result:
                score_dict['presence_probability'] = result['presence_probability']
            if 'intensity_score' in result:
                score_dict['intensity_score'] = result['intensity_score']
            if 'pdq_s_score' in result:
                score_dict['pdq_s_score'] = result['pdq_s_score']
            if 'total' in result:
                score_dict['total'] = result['total']
            
            # If no specific fields, try to extract any numeric values
            if not score_dict:
                for key, value in result.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        score_dict[key] = float(value)
                        break  # Use first numeric value found
        
        if score_dict:
            extracted.append(score_dict)
        elif isinstance(result, (int, float)):
            extracted.append({'score': float(result)})
    
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Validate PDQ-S metric using permutation test"
    )
    parser.add_argument(
        "--lsd-results",
        type=str,
        required=True,
        help="Path to JSON file containing LSD PDQ-S results"
    )
    parser.add_argument(
        "--placebo-results",
        type=str,
        required=True,
        help="Path to JSON file containing Placebo PDQ-S results"
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of permutations (default: 10000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save validation results JSON (optional)"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save null distribution plot (optional)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading LSD results from {args.lsd_results}")
    lsd_data = load_results_from_file(Path(args.lsd_results))
    lsd_results = extract_pdq_s_scores(lsd_data)
    
    logger.info(f"Loading Placebo results from {args.placebo_results}")
    placebo_data = load_results_from_file(Path(args.placebo_results))
    placebo_results = extract_pdq_s_scores(placebo_data)
    
    logger.info(f"Loaded {len(lsd_results)} LSD results and {len(placebo_results)} Placebo results")
    
    if len(lsd_results) == 0 or len(placebo_results) == 0:
        logger.error("Insufficient results loaded. Need at least one result for each condition.")
        return 1
    
    # Run permutation test
    logger.info(f"Running permutation test with {args.n_permutations} permutations...")
    result = validate_pdq_s_metric(
        lsd_results=lsd_results,
        placebo_results=placebo_results,
        n_permutations=args.n_permutations
    )
    
    # Print results
    print("\n" + "="*60)
    print("PDQ-S METRIC VALIDATION RESULTS")
    print("="*60)
    print(f"Metric: {result.metric_name}")
    print(f"Actual Score Difference: {result.actual_score:.4f}")
    print(f"Null Distribution:")
    print(f"  Mean: {result.null_mean:.4f}")
    print(f"  Std:  {result.null_std:.4f}")
    print(f"  Median: {result.null_median:.4f}")
    print(f"\nP-value: {result.p_value:.4f}")
    print(f"Significant (p < 0.01): {result.significant}")
    print(f"\nInterpretation:")
    print(f"  {result.interpretation}")
    print("="*60)
    
    # Save results
    if args.output:
        output_data = {
            'metric_name': result.metric_name,
            'actual_score': float(result.actual_score),
            'null_mean': float(result.null_mean),
            'null_std': float(result.null_std),
            'null_median': float(result.null_median),
            'p_value': float(result.p_value),
            'n_permutations': result.n_permutations,
            'significant': result.significant,
            'interpretation': result.interpretation,
            'n_lsd': len(lsd_results),
            'n_placebo': len(placebo_results)
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved validation results to {args.output}")
    
    # Create plot if requested
    if args.plot:
        validator = PermutationTestValidator(n_permutations=args.n_permutations, random_seed=args.random_seed)
        validator.plot_null_distribution(result, save_path=args.plot)
    
    # Return exit code based on significance
    return 0 if result.significant else 1


if __name__ == "__main__":
    sys.exit(main())

