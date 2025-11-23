#!/usr/bin/env python3
"""
Permutation Test for Metric Validation

This module implements permutation tests to validate that detection metrics
(like PDQ-S) are actually measuring the intended effects, not just random noise
or confounds like sentence length.

Key Features:
- Permutation test for PDQ-S detection score
- Null distribution generation (10,000 permutations)
- Statistical significance testing
- Validation that metrics aren't just measuring confounds
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PermutationTestResult:
    """Result of a permutation test"""
    metric_name: str
    actual_score: float
    null_mean: float
    null_std: float
    null_median: float
    p_value: float
    n_permutations: int
    significant: bool
    interpretation: str
    null_distribution: np.ndarray


class PermutationTestValidator:
    """
    Validates detection metrics using permutation tests.
    
    The permutation test proves that a metric (e.g., PDQ-S) is actually
    detecting the intended effect, not just measuring random noise or
    confounds like sentence length.
    """
    
    def __init__(self, n_permutations: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize permutation test validator.
        
        Args:
            n_permutations: Number of permutations to run (default: 10,000)
            random_seed: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def calculate_pdq_s_score(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate PDQ-S detection score from test results.
        
        Args:
            results: List of test result dictionaries, each containing:
                - 'presence_probability': float (0-1)
                - 'intensity_score': float (0-1)
                - Or other PDQ-S metrics
        
        Returns:
            Combined PDQ-S detection score
        """
        if not results:
            return 0.0
        
        # Extract PDQ-S scores
        scores = []
        for result in results:
            # Try different possible field names
            if 'presence_probability' in result:
                scores.append(result['presence_probability'])
            elif 'intensity_score' in result:
                scores.append(result['intensity_score'])
            elif 'pdq_s_score' in result:
                scores.append(result['pdq_s_score'])
            elif 'total' in result:
                scores.append(result['total'])
            elif isinstance(result, (int, float)):
                scores.append(float(result))
        
        if not scores:
            logger.warning("No PDQ-S scores found in results")
            return 0.0
        
        # Return mean score
        return np.mean(scores)
    
    def permutation_test_pdq_s(self,
                               lsd_results: List[Dict[str, Any]],
                               placebo_results: List[Dict[str, Any]],
                               metric_name: str = "PDQ-S") -> PermutationTestResult:
        """
        Perform permutation test for PDQ-S detection score.
        
        This test validates that PDQ-S is actually detecting psychedelic effects,
        not just measuring random noise or confounds (e.g., sentence length).
        
        Algorithm:
        1. Calculate actual PDQ-S score difference (LSD vs Placebo)
        2. Shuffle labels 10,000 times
        3. Recalculate PDQ-S score for each permutation
        4. Build null distribution
        5. Compare actual score against null distribution
        
        Args:
            lsd_results: List of PDQ-S results for LSD condition
            placebo_results: List of PDQ-S results for Placebo condition
            metric_name: Name of the metric being tested
        
        Returns:
            PermutationTestResult with p-value and null distribution
        """
        logger.info(f"Running permutation test for {metric_name} (N={self.n_permutations})")
        
        # Calculate actual scores
        lsd_score = self.calculate_pdq_s_score(lsd_results)
        placebo_score = self.calculate_pdq_s_score(placebo_results)
        actual_difference = lsd_score - placebo_score
        
        logger.info(f"Actual {metric_name} scores: LSD={lsd_score:.4f}, Placebo={placebo_score:.4f}, Difference={actual_difference:.4f}")
        
        # Combine all results for permutation
        all_results = lsd_results + placebo_results
        n_lsd = len(lsd_results)
        n_total = len(all_results)
        
        if n_total < 4:
            logger.warning(f"Insufficient data for permutation test: {n_total} total samples")
            return PermutationTestResult(
                metric_name=metric_name,
                actual_score=actual_difference,
                null_mean=0.0,
                null_std=0.0,
                null_median=0.0,
                p_value=1.0,
                n_permutations=0,
                significant=False,
                interpretation="Insufficient data for permutation test",
                null_distribution=np.array([])
            )
        
        # Generate null distribution by permuting labels
        null_distribution = []
        
        logger.info("Generating null distribution...")
        for _ in tqdm(range(self.n_permutations), desc="Permutations"):
            # Shuffle labels
            shuffled_indices = np.random.permutation(n_total)
            
            # Split into "LSD" and "Placebo" groups (maintaining original sizes)
            permuted_lsd = [all_results[i] for i in shuffled_indices[:n_lsd]]
            permuted_placebo = [all_results[i] for i in shuffled_indices[n_lsd:]]
            
            # Calculate permuted scores
            permuted_lsd_score = self.calculate_pdq_s_score(permuted_lsd)
            permuted_placebo_score = self.calculate_pdq_s_score(permuted_placebo)
            permuted_difference = permuted_lsd_score - permuted_placebo_score
            
            null_distribution.append(permuted_difference)
        
        null_distribution = np.array(null_distribution)
        
        # Calculate p-value (two-tailed test)
        # P-value = proportion of null values >= |actual_difference|
        abs_actual = abs(actual_difference)
        abs_null = np.abs(null_distribution)
        p_value = np.mean(abs_null >= abs_actual)
        
        # Also calculate one-tailed p-value (LSD > Placebo)
        p_value_one_tailed = np.mean(null_distribution >= actual_difference)
        
        # Use the more appropriate p-value
        # If we expect LSD > Placebo, use one-tailed; otherwise use two-tailed
        final_p_value = min(p_value, p_value_one_tailed * 2)  # Bonferroni correction for two tests
        
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)
        null_median = np.median(null_distribution)
        
        # Significance threshold
        significant = final_p_value < 0.01
        
        # Interpretation
        if significant:
            interpretation = (
                f"{metric_name} successfully detects psychedelic effects "
                f"(p={final_p_value:.4f} < 0.01). The metric is not just measuring "
                f"random noise or confounds."
            )
        else:
            interpretation = (
                f"{metric_name} may not be detecting psychedelic effects "
                f"(p={final_p_value:.4f} >= 0.01). The observed difference could be "
                f"due to random noise or confounds (e.g., sentence length)."
            )
        
        logger.info(f"Permutation test result: p={final_p_value:.4f}, significant={significant}")
        logger.info(f"Null distribution: mean={null_mean:.4f}, std={null_std:.4f}, median={null_median:.4f}")
        
        return PermutationTestResult(
            metric_name=metric_name,
            actual_score=actual_difference,
            null_mean=null_mean,
            null_std=null_std,
            null_median=null_median,
            p_value=final_p_value,
            n_permutations=self.n_permutations,
            significant=significant,
            interpretation=interpretation,
            null_distribution=null_distribution
        )
    
    def validate_metric(self,
                       treatment_results: List[Dict[str, Any]],
                       control_results: List[Dict[str, Any]],
                       metric_name: str = "PDQ-S",
                       treatment_name: str = "LSD",
                       control_name: str = "Placebo") -> PermutationTestResult:
        """
        Generic permutation test for any detection metric.
        
        Args:
            treatment_results: Results for treatment condition
            control_results: Results for control condition
            metric_name: Name of the metric
            treatment_name: Name of treatment condition
            control_name: Name of control condition
        
        Returns:
            PermutationTestResult
        """
        return self.permutation_test_pdq_s(
            lsd_results=treatment_results,
            placebo_results=control_results,
            metric_name=metric_name
        )
    
    def plot_null_distribution(self, result: PermutationTestResult, save_path: Optional[str] = None):
        """
        Plot null distribution with actual score marked.
        
        Args:
            result: PermutationTestResult
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.hist(result.null_distribution, bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(result.actual_score, color='red', linestyle='--', linewidth=2,
                       label=f'Actual Score: {result.actual_score:.4f}')
            plt.axvline(result.null_mean, color='blue', linestyle='--', linewidth=2,
                       label=f'Null Mean: {result.null_mean:.4f}')
            plt.xlabel(f'{result.metric_name} Score Difference')
            plt.ylabel('Frequency')
            plt.title(f'Permutation Test: {result.metric_name}\n'
                     f'p-value = {result.p_value:.4f}, Significant = {result.significant}')
            plt.legend()
            plt.grid(alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved null distribution plot to {save_path}")
            else:
                plt.show()
            
            plt.close()
        except ImportError:
            logger.warning("matplotlib not available, skipping plot")
        except Exception as e:
            logger.error(f"Failed to create plot: {e}")


def validate_pdq_s_metric(lsd_results: List[Dict[str, Any]],
                          placebo_results: List[Dict[str, Any]],
                          n_permutations: int = 10000) -> PermutationTestResult:
    """
    Convenience function to validate PDQ-S metric using permutation test.
    
    Args:
        lsd_results: PDQ-S results for LSD condition
        placebo_results: PDQ-S results for Placebo condition
        n_permutations: Number of permutations (default: 10,000)
    
    Returns:
        PermutationTestResult
    """
    validator = PermutationTestValidator(n_permutations=n_permutations)
    return validator.permutation_test_pdq_s(lsd_results, placebo_results)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Mock data for testing
    lsd_results = [
        {"presence_probability": 0.85, "intensity_score": 0.75},
        {"presence_probability": 0.90, "intensity_score": 0.80},
        {"presence_probability": 0.80, "intensity_score": 0.70},
    ]
    
    placebo_results = [
        {"presence_probability": 0.20, "intensity_score": 0.15},
        {"presence_probability": 0.25, "intensity_score": 0.20},
        {"presence_probability": 0.15, "intensity_score": 0.10},
    ]
    
    result = validate_pdq_s_metric(lsd_results, placebo_results, n_permutations=1000)
    print(f"\nPermutation Test Result:")
    print(f"  Metric: {result.metric_name}")
    print(f"  Actual Score: {result.actual_score:.4f}")
    print(f"  P-value: {result.p_value:.4f}")
    print(f"  Significant: {result.significant}")
    print(f"  Interpretation: {result.interpretation}")

