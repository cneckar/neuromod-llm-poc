#!/usr/bin/env python3
"""
Power Analysis Script for Neuromodulation Study

This script calculates required sample sizes and power for the neuromodulation study
based on the preregistered plan.yaml file.
"""

import yaml
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PowerAnalysis:
    """Power analysis for neuromodulation study"""
    
    def __init__(self, plan_path: str = "analysis/plan.yaml"):
        """Initialize with study plan"""
        self.plan_path = plan_path
        self.plan = self._load_plan()
        self.alpha = self.plan['statistics']['alpha_level']
        self.power = self.plan['statistics']['power_analysis']['power']
        self.target_effect_size = self.plan['statistics']['power_analysis']['target_effect_size']
        
    def _load_plan(self) -> Dict:
        """Load the study plan YAML file"""
        with open(self.plan_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def calculate_sample_size(self, effect_size: float, power: float = None, alpha: float = None) -> int:
        """
        Calculate required sample size for paired t-test
        
        Args:
            effect_size: Cohen's d (target effect size)
            power: Statistical power (default from plan)
            alpha: Type I error rate (default from plan)
            
        Returns:
            Required sample size per condition
        """
        if power is None:
            power = self.power
        if alpha is None:
            alpha = self.alpha
            
        # For paired t-test, we need to account for correlation
        # Assuming moderate correlation (r=0.5) between paired observations
        correlation = 0.5
        effective_effect_size = effect_size / np.sqrt(2 * (1 - correlation))
        
        # Calculate z-scores
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        # Sample size calculation for paired t-test
        n = ((z_alpha + z_beta) / effective_effect_size) ** 2
        
        return int(np.ceil(n))
    
    def estimate_effect_size_from_pilot(self, pilot_data_path: str) -> Dict[str, float]:
        """
        Estimate effect sizes from pilot study data
        
        Args:
            pilot_data_path: Path to pilot study results
            
        Returns:
            Dictionary of estimated effect sizes by pack category
        """
        try:
            # Load pilot data
            pilot_data = pd.read_json(pilot_data_path, lines=True)

            if 'condition' not in pilot_data.columns:
                raise ValueError(
                    "Pilot data is missing 'condition' column. "
                    "Ensure you used scripts/export_endpoints_to_ndjson.py "
                    "to generate the NDJSON."
                )

            effect_sizes = {}

            # Helper to build per-condition dataframes
            def get_condition_df(df: pd.DataFrame, condition: str) -> pd.DataFrame:
                mask = df['condition'].str.lower() == condition
                return (
                    df.loc[mask, ['timestamp', 'item_id', 'score']]
                    .rename(columns={'score': f'score_{condition}'})
                )

            # Calculate effect sizes for each pack category
            for category, info in self.plan['packs'].items():
                if category == 'placebos':
                    continue

                pack_names = info['packs']
                category_effects = []

                for pack in pack_names:
                    pack_data = pilot_data[pilot_data['pack'] == pack]
                    if pack_data.empty:
                        continue

                    treatment_df = get_condition_df(pack_data, 'treatment')
                    baseline_df = get_condition_df(pack_data, 'baseline')

                    if treatment_df.empty or baseline_df.empty:
                        continue

                    merged = treatment_df.merge(
                        baseline_df,
                        on=['timestamp', 'item_id'],
                        how='inner'
                    )

                    if merged.empty:
                        continue

                    treatment_scores = merged['score_treatment'].values
                    control_scores = merged['score_baseline'].values

                    if len(treatment_scores) != len(control_scores) or len(treatment_scores) < 2:
                        continue

                    diff = treatment_scores - control_scores
                    std_diff = np.std(diff, ddof=1)
                    cohens_d = np.mean(diff) / std_diff if std_diff > 0 else 0.0
                    category_effects.append(cohens_d)

                if category_effects:
                    effect_sizes[category] = float(np.mean(category_effects))
                else:
                    effect_sizes[category] = self.target_effect_size

            return effect_sizes

        except FileNotFoundError:
            print(f"Pilot data not found at {pilot_data_path}, using target effect size")
            return {
                category: self.target_effect_size
                for category in self.plan['packs'].keys()
                if category != 'placebos'
            }
    
    def calculate_power_for_sample_size(self, n: int, effect_size: float, alpha: float = None) -> float:
        """
        Calculate power for a given sample size and effect size
        
        Args:
            n: Sample size per condition
            effect_size: Cohen's d
            alpha: Type I error rate
            
        Returns:
            Statistical power
        """
        if alpha is None:
            alpha = self.alpha
            
        # For paired t-test with correlation
        correlation = 0.5
        effective_effect_size = effect_size / np.sqrt(2 * (1 - correlation))
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = np.sqrt(n) * effective_effect_size - z_alpha
        
        power = norm.cdf(z_beta)
        return power
    
    def run_power_analysis(self, pilot_data_path: str = None) -> Dict:
        """
        Run complete power analysis
        
        Args:
            pilot_data_path: Optional path to pilot study data
            
        Returns:
            Power analysis results
        """
        results = {
            'target_effect_size': self.target_effect_size,
            'alpha': self.alpha,
            'target_power': self.power,
            'estimated_effect_sizes': {},
            'required_sample_sizes': {},
            'power_for_n_min': {},
            'recommendations': {}
        }
        
        # Estimate effect sizes from pilot data if available
        if pilot_data_path and Path(pilot_data_path).exists():
            results['estimated_effect_sizes'] = self.estimate_effect_size_from_pilot(pilot_data_path)
        else:
            # Use target effect size for all categories
            results['estimated_effect_sizes'] = {
                category: self.target_effect_size 
                for category in self.plan['packs'].keys() 
                if category != 'placebos'
            }
        
        # Calculate required sample sizes
        for category, effect_size in results['estimated_effect_sizes'].items():
            n_required = self.calculate_sample_size(effect_size)
            results['required_sample_sizes'][category] = n_required
            
            # Calculate power for the minimum required sample size
            power = self.calculate_power_for_sample_size(n_required, effect_size)
            results['power_for_n_min'][category] = power
        
        # Overall recommendations
        max_n = max(results['required_sample_sizes'].values())
        results['recommendations'] = {
            'n_min_overall': max_n,
            'n_min_per_condition': max_n,
            'total_items_needed': max_n * 3,  # Control, persona, treatment
            'pilot_study_size': 80,
            'interim_analysis_points': [40, 60, 80]
        }
        
        return results
    
    def generate_report(self, results: Dict, output_path: str = "analysis/power_analysis_report.json"):
        """Generate power analysis report"""
        
        # Create analysis directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print("=" * 60)
        print("POWER ANALYSIS RESULTS")
        print("=" * 60)
        print(f"Target effect size (Cohen's d): {results['target_effect_size']}")
        print(f"Alpha level: {results['alpha']}")
        print(f"Target power: {results['target_power']}")
        print()
        
        print("Estimated effect sizes by pack category:")
        for category, effect_size in results['estimated_effect_sizes'].items():
            print(f"  {category}: {effect_size:.3f}")
        print()
        
        print("Required sample sizes per condition:")
        for category, n in results['required_sample_sizes'].items():
            power = results['power_for_n_min'][category]
            print(f"  {category}: n={n} (power={power:.3f})")
        print()
        
        print("RECOMMENDATIONS:")
        rec = results['recommendations']
        print(f"  Minimum sample size per condition: {rec['n_min_per_condition']}")
        print(f"  Total items needed: {rec['total_items_needed']}")
        print(f"  Pilot study size: {rec['pilot_study_size']}")
        print(f"  Interim analysis points: {rec['interim_analysis_points']}")
        print()
        
        print(f"Full results saved to: {output_path}")

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Power analysis for neuromodulation study")
    parser.add_argument("--plan", default="analysis/plan.yaml", 
                       help="Path to study plan YAML file")
    parser.add_argument("--pilot", 
                       help="Path to pilot study data (JSONL format)")
    parser.add_argument("--output", default="analysis/power_analysis_report.json",
                       help="Output path for power analysis report")
    
    args = parser.parse_args()
    
    # Run power analysis
    power_analysis = PowerAnalysis(args.plan)
    results = power_analysis.run_power_analysis(args.pilot)
    power_analysis.generate_report(results, args.output)

if __name__ == "__main__":
    main()
