"""
Ablations and Dose-response Analysis System

This module implements systematic ablation studies and dose-response analysis
for neuromodulation effects, enabling component-wise analysis and effect
magnitude studies.

Key Features:
- Component ablation analysis (remove individual effects)
- Dose-response curves (vary effect weights)
- Interaction analysis (effect combinations)
- Statistical validation of ablations
- Automated ablation report generation
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AblationResult:
    """Results from a single ablation experiment"""
    effect_name: str
    effect_removed: bool
    original_score: float
    ablated_score: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    sample_size: int

@dataclass
class DoseResponsePoint:
    """Single point on a dose-response curve"""
    dose: float  # Effect weight/magnitude
    response: float  # Measured metric
    n_samples: int
    std_error: float
    confidence_interval: Tuple[float, float]

@dataclass
class DoseResponseCurve:
    """Complete dose-response curve analysis"""
    effect_name: str
    metric_name: str
    doses: List[float]
    responses: List[float]
    curve_params: Dict[str, float]
    r_squared: float
    ec50: Optional[float]  # 50% effective concentration
    hill_slope: Optional[float]
    max_response: float
    min_response: float
    curve_type: str  # 'sigmoid', 'linear', 'exponential', 'polynomial'

@dataclass
class InteractionAnalysis:
    """Analysis of effect interactions"""
    effect_pair: Tuple[str, str]
    individual_effects: Tuple[float, float]
    combined_effect: float
    interaction_effect: float
    interaction_p_value: float
    synergy_score: float  # >1 = synergy, <1 = antagonism, =1 = additive
    confidence_interval: Tuple[float, float]

class AblationsAnalyzer:
    """Main class for conducting ablation and dose-response analysis"""
    
    def __init__(self, results_dir: str = "analysis/ablations"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.ablation_results: List[AblationResult] = []
        self.dose_response_curves: List[DoseResponseCurve] = []
        self.interaction_analyses: List[InteractionAnalysis] = []
        
    def run_component_ablations(self, 
                              pack_configs: Dict[str, Dict],
                              baseline_results: Dict[str, float],
                              test_data: List[Dict],
                              model_backend: str = "huggingface") -> List[AblationResult]:
        """
        Run systematic ablation studies by removing individual effects
        
        Args:
            pack_configs: Dictionary of pack configurations
            baseline_results: Baseline performance metrics
            test_data: Test dataset
            model_backend: Model backend to use
            
        Returns:
            List of ablation results
        """
        logger.info("Starting component ablation analysis...")
        
        ablation_results = []
        
        for pack_name, pack_config in pack_configs.items():
            if pack_name in ['none', 'placebo']:
                continue  # Skip control packs
                
            logger.info(f"Analyzing pack: {pack_name}")
            
            # Get baseline performance for this pack
            baseline_score = baseline_results.get(f"{pack_name}_score", 0.0)
            
            # Get list of effects in this pack
            effects = pack_config.get('effects', [])
            
            for effect in effects:
                effect_name = effect.get('effect', 'unknown')
                logger.info(f"  Ablating effect: {effect_name}")
                
                # Create ablated pack (remove this effect)
                ablated_config = self._create_ablated_pack(pack_config, effect_name)
                
                # Run test with ablated pack
                ablated_score = self._run_ablated_test(
                    ablated_config, test_data, model_backend
                )
                
                # Calculate effect size and statistics
                effect_size = baseline_score - ablated_score
                p_value, ci = self._calculate_ablation_stats(
                    baseline_score, ablated_score, len(test_data)
                )
                
                # Store result
                result = AblationResult(
                    effect_name=effect_name,
                    effect_removed=True,
                    original_score=baseline_score,
                    ablated_score=ablated_score,
                    effect_size=effect_size,
                    p_value=p_value,
                    confidence_interval=ci,
                    sample_size=len(test_data)
                )
                
                ablation_results.append(result)
                self.ablation_results.append(result)
                
                logger.info(f"    Effect size: {effect_size:.4f}, p-value: {p_value:.4f}")
        
        # Save results
        self._save_ablation_results(ablation_results)
        
        return ablation_results
    
    def run_dose_response_analysis(self,
                                 effect_name: str,
                                 base_pack_config: Dict,
                                 dose_range: List[float],
                                 test_data: List[Dict],
                                 metric_name: str = "adq_score",
                                 model_backend: str = "huggingface") -> DoseResponseCurve:
        """
        Generate dose-response curve for a specific effect
        
        Args:
            effect_name: Name of effect to vary
            base_pack_config: Base pack configuration
            dose_range: List of dose values (effect weights)
            test_data: Test dataset
            metric_name: Metric to measure
            model_backend: Model backend to use
            
        Returns:
            Dose-response curve analysis
        """
        logger.info(f"Generating dose-response curve for {effect_name}")
        
        doses = []
        responses = []
        std_errors = []
        confidence_intervals = []
        
        for dose in dose_range:
            logger.info(f"  Testing dose: {dose}")
            
            # Create pack with modified effect weight
            dosed_config = self._create_dosed_pack(base_pack_config, effect_name, dose)
            
            # Run test
            response = self._run_dosed_test(
                dosed_config, test_data, model_backend, metric_name
            )
            
            # Calculate statistics
            std_error = self._calculate_std_error(response, len(test_data))
            ci = self._calculate_confidence_interval(response, std_error, len(test_data))
            
            doses.append(dose)
            responses.append(response)
            std_errors.append(std_error)
            confidence_intervals.append(ci)
        
        # Fit dose-response curve
        curve_params, r_squared, curve_type = self._fit_dose_response_curve(
            doses, responses
        )
        
        # Calculate EC50 and other parameters
        ec50, hill_slope = self._calculate_ec50_and_hill_slope(
            doses, responses, curve_params, curve_type
        )
        
        # Create dose-response curve
        curve = DoseResponseCurve(
            effect_name=effect_name,
            metric_name=metric_name,
            doses=doses,
            responses=responses,
            curve_params=curve_params,
            r_squared=r_squared,
            ec50=ec50,
            hill_slope=hill_slope,
            max_response=max(responses),
            min_response=min(responses),
            curve_type=curve_type
        )
        
        self.dose_response_curves.append(curve)
        
        # Save results
        self._save_dose_response_curve(curve)
        
        logger.info(f"  EC50: {ec50}, R²: {r_squared:.4f}, Curve type: {curve_type}")
        
        return curve
    
    def analyze_effect_interactions(self,
                                  pack_configs: Dict[str, Dict],
                                  test_data: List[Dict],
                                  model_backend: str = "huggingface") -> List[InteractionAnalysis]:
        """
        Analyze interactions between different effects
        
        Args:
            pack_configs: Dictionary of pack configurations
            test_data: Test dataset
            model_backend: Model backend to use
            
        Returns:
            List of interaction analyses
        """
        logger.info("Analyzing effect interactions...")
        
        interaction_analyses = []
        
        # Get all unique effects across packs
        all_effects = set()
        for pack_config in pack_configs.values():
            for effect in pack_config.get('effects', []):
                all_effects.add(effect.get('effect', 'unknown'))
        
        all_effects = list(all_effects)
        
        # Test all pairwise combinations
        for i, effect1 in enumerate(all_effects):
            for effect2 in all_effects[i+1:]:
                logger.info(f"  Testing interaction: {effect1} × {effect2}")
                
                # Create packs with individual effects
                pack1 = self._create_single_effect_pack(effect1)
                pack2 = self._create_single_effect_pack(effect2)
                pack_combined = self._create_combined_effect_pack(effect1, effect2)
                
                # Run tests
                score1 = self._run_dosed_test(pack1, test_data, model_backend)
                score2 = self._run_dosed_test(pack2, test_data, model_backend)
                score_combined = self._run_dosed_test(pack_combined, test_data, model_backend)
                
                # Calculate interaction effect
                expected_additive = score1 + score2
                interaction_effect = score_combined - expected_additive
                synergy_score = score_combined / expected_additive if expected_additive != 0 else 1.0
                
                # Statistical test
                p_value, ci = self._calculate_interaction_stats(
                    score1, score2, score_combined, len(test_data)
                )
                
                # Store result
                analysis = InteractionAnalysis(
                    effect_pair=(effect1, effect2),
                    individual_effects=(score1, score2),
                    combined_effect=score_combined,
                    interaction_effect=interaction_effect,
                    interaction_p_value=p_value,
                    synergy_score=synergy_score,
                    confidence_interval=ci
                )
                
                interaction_analyses.append(analysis)
                self.interaction_analyses.append(analysis)
                
                logger.info(f"    Synergy score: {synergy_score:.4f}, p-value: {p_value:.4f}")
        
        # Save results
        self._save_interaction_analyses(interaction_analyses)
        
        return interaction_analyses
    
    def generate_ablation_report(self) -> Dict[str, Any]:
        """Generate comprehensive ablation analysis report"""
        logger.info("Generating ablation analysis report...")
        
        report = {
            "summary": {
                "total_ablations": len(self.ablation_results),
                "total_dose_response_curves": len(self.dose_response_curves),
                "total_interactions": len(self.interaction_analyses),
                "timestamp": pd.Timestamp.now().isoformat()
            },
            "ablation_results": [asdict(result) for result in self.ablation_results],
            "dose_response_curves": [asdict(curve) for curve in self.dose_response_curves],
            "interaction_analyses": [asdict(analysis) for analysis in self.interaction_analyses],
            "statistical_summary": self._generate_statistical_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        # Save report
        report_path = self.results_dir / "ablation_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Ablation report saved to: {report_path}")
        
        return report
    
    def _create_ablated_pack(self, pack_config: Dict, effect_to_remove: str) -> Dict:
        """Create pack configuration with specific effect removed"""
        ablated_config = pack_config.copy()
        ablated_config['effects'] = [
            effect for effect in pack_config.get('effects', [])
            if effect.get('effect') != effect_to_remove
        ]
        return ablated_config
    
    def _create_dosed_pack(self, pack_config: Dict, effect_name: str, dose: float) -> Dict:
        """Create pack configuration with modified effect weight"""
        dosed_config = pack_config.copy()
        dosed_config['effects'] = []
        
        for effect in pack_config.get('effects', []):
            if effect.get('effect') == effect_name:
                # Modify the weight
                modified_effect = effect.copy()
                modified_effect['weight'] = dose
                dosed_config['effects'].append(modified_effect)
            else:
                dosed_config['effects'].append(effect)
        
        return dosed_config
    
    def _create_single_effect_pack(self, effect_name: str) -> Dict:
        """Create pack with single effect"""
        return {
            "name": f"single_{effect_name}",
            "effects": [{"effect": effect_name, "weight": 1.0, "direction": "up"}]
        }
    
    def _create_combined_effect_pack(self, effect1: str, effect2: str) -> Dict:
        """Create pack with two effects combined"""
        return {
            "name": f"combined_{effect1}_{effect2}",
            "effects": [
                {"effect": effect1, "weight": 1.0, "direction": "up"},
                {"effect": effect2, "weight": 1.0, "direction": "up"}
            ]
        }
    
    def _run_ablated_test(self, pack_config: Dict, test_data: List[Dict], 
                         model_backend: str) -> float:
        """Run test with ablated pack configuration"""
        # This would integrate with the actual testing framework
        # For now, return a mock score
        return np.random.normal(0.5, 0.1)
    
    def _run_dosed_test(self, pack_config: Dict, test_data: List[Dict], 
                       model_backend: str, metric_name: str = "adq_score") -> float:
        """Run test with dosed pack configuration"""
        # This would integrate with the actual testing framework
        # For now, return a mock score
        return np.random.normal(0.5, 0.1)
    
    def _calculate_ablation_stats(self, original: float, ablated: float, 
                                n_samples: int) -> Tuple[float, Tuple[float, float]]:
        """Calculate statistics for ablation comparison"""
        # Paired t-test
        t_stat, p_value = stats.ttest_1samp([original - ablated], 0)
        
        # Effect size (Cohen's d)
        effect_size = (original - ablated) / 0.1  # Assuming std = 0.1
        
        # Confidence interval
        se = 0.1 / np.sqrt(n_samples)  # Standard error
        ci = (effect_size - 1.96 * se, effect_size + 1.96 * se)
        
        return p_value, ci
    
    def _calculate_std_error(self, response: float, n_samples: int) -> float:
        """Calculate standard error for response"""
        return 0.1 / np.sqrt(n_samples)  # Mock calculation
    
    def _calculate_confidence_interval(self, response: float, std_error: float, 
                                     n_samples: int) -> Tuple[float, float]:
        """Calculate confidence interval for response"""
        ci = (response - 1.96 * std_error, response + 1.96 * std_error)
        return ci
    
    def _fit_dose_response_curve(self, doses: List[float], 
                               responses: List[float]) -> Tuple[Dict, float, str]:
        """Fit dose-response curve and determine best model"""
        doses = np.array(doses)
        responses = np.array(responses)
        
        # Try different curve types
        curve_types = ['sigmoid', 'linear', 'exponential', 'polynomial']
        best_r2 = -np.inf
        best_params = {}
        best_type = 'linear'
        
        for curve_type in curve_types:
            try:
                if curve_type == 'sigmoid':
                    # Sigmoid: y = a / (1 + exp(-b * (x - c)))
                    def sigmoid(x, a, b, c):
                        return a / (1 + np.exp(-b * (x - c)))
                    
                    popt, _ = curve_fit(sigmoid, doses, responses, maxfev=1000)
                    predicted = sigmoid(doses, *popt)
                    r2 = 1 - np.sum((responses - predicted) ** 2) / np.sum((responses - np.mean(responses)) ** 2)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
                        best_type = 'sigmoid'
                
                elif curve_type == 'linear':
                    # Linear: y = a * x + b
                    popt = np.polyfit(doses, responses, 1)
                    predicted = np.polyval(popt, doses)
                    r2 = 1 - np.sum((responses - predicted) ** 2) / np.sum((responses - np.mean(responses)) ** 2)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {'slope': popt[0], 'intercept': popt[1]}
                        best_type = 'linear'
                
                elif curve_type == 'exponential':
                    # Exponential: y = a * exp(b * x)
                    def exponential(x, a, b):
                        return a * np.exp(b * x)
                    
                    popt, _ = curve_fit(exponential, doses, responses, maxfev=1000)
                    predicted = exponential(doses, *popt)
                    r2 = 1 - np.sum((responses - predicted) ** 2) / np.sum((responses - np.mean(responses)) ** 2)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {'a': popt[0], 'b': popt[1]}
                        best_type = 'exponential'
                
                elif curve_type == 'polynomial':
                    # Polynomial: y = a * x^2 + b * x + c
                    popt = np.polyfit(doses, responses, 2)
                    predicted = np.polyval(popt, doses)
                    r2 = 1 - np.sum((responses - predicted) ** 2) / np.sum((responses - np.mean(responses)) ** 2)
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_params = {'a': popt[0], 'b': popt[1], 'c': popt[2]}
                        best_type = 'polynomial'
                        
            except Exception as e:
                logger.warning(f"Failed to fit {curve_type} curve: {e}")
                continue
        
        return best_params, best_r2, best_type
    
    def _calculate_ec50_and_hill_slope(self, doses: List[float], responses: List[float],
                                      curve_params: Dict, curve_type: str) -> Tuple[Optional[float], Optional[float]]:
        """Calculate EC50 and Hill slope for dose-response curve"""
        if curve_type == 'sigmoid':
            # For sigmoid: EC50 = c, Hill slope = b
            ec50 = curve_params.get('c')
            hill_slope = curve_params.get('b')
            return ec50, hill_slope
        else:
            # For other curve types, EC50 not well-defined
            return None, None
    
    def _calculate_interaction_stats(self, score1: float, score2: float, 
                                   score_combined: float, n_samples: int) -> Tuple[float, Tuple[float, float]]:
        """Calculate statistics for interaction analysis"""
        # Simple t-test for interaction effect
        interaction_effect = score_combined - (score1 + score2)
        t_stat, p_value = stats.ttest_1samp([interaction_effect], 0)
        
        # Confidence interval
        se = 0.1 / np.sqrt(n_samples)
        ci = (interaction_effect - 1.96 * se, interaction_effect + 1.96 * se)
        
        return p_value, ci
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of all analyses"""
        if not self.ablation_results:
            return {"message": "No ablation results available"}
        
        # Ablation summary
        significant_ablations = [r for r in self.ablation_results if r.p_value < 0.05]
        effect_sizes = [r.effect_size for r in self.ablation_results]
        
        # Dose-response summary
        significant_curves = [c for c in self.dose_response_curves if c.r_squared > 0.7]
        
        # Interaction summary
        synergistic_interactions = [i for i in self.interaction_analyses if i.synergy_score > 1.1]
        antagonistic_interactions = [i for i in self.interaction_analyses if i.synergy_score < 0.9]
        
        return {
            "ablation_summary": {
                "total_ablations": len(self.ablation_results),
                "significant_ablations": len(significant_ablations),
                "mean_effect_size": np.mean(effect_sizes),
                "std_effect_size": np.std(effect_sizes),
                "largest_effect": max(effect_sizes) if effect_sizes else 0,
                "smallest_effect": min(effect_sizes) if effect_sizes else 0
            },
            "dose_response_summary": {
                "total_curves": len(self.dose_response_curves),
                "significant_curves": len(significant_curves),
                "mean_r_squared": np.mean([c.r_squared for c in self.dose_response_curves]) if self.dose_response_curves else 0,
                "curves_with_ec50": len([c for c in self.dose_response_curves if c.ec50 is not None])
            },
            "interaction_summary": {
                "total_interactions": len(self.interaction_analyses),
                "synergistic_interactions": len(synergistic_interactions),
                "antagonistic_interactions": len(antagonistic_interactions),
                "mean_synergy_score": np.mean([i.synergy_score for i in self.interaction_analyses]) if self.interaction_analyses else 1.0
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        if not self.ablation_results:
            return ["No ablation results available for recommendations"]
        
        # Analyze effect sizes
        large_effects = [r for r in self.ablation_results if abs(r.effect_size) > 0.5]
        if large_effects:
            recommendations.append(f"Found {len(large_effects)} effects with large impact (|effect size| > 0.5)")
        
        # Analyze dose-response curves
        if self.dose_response_curves:
            high_r2_curves = [c for c in self.dose_response_curves if c.r_squared > 0.8]
            if high_r2_curves:
                recommendations.append(f"Found {len(high_r2_curves)} dose-response curves with excellent fit (R² > 0.8)")
        
        # Analyze interactions
        if self.interaction_analyses:
            strong_synergies = [i for i in self.interaction_analyses if i.synergy_score > 1.5]
            if strong_synergies:
                recommendations.append(f"Found {len(strong_synergies)} strong synergistic interactions")
        
        return recommendations
    
    def _save_ablation_results(self, results: List[AblationResult]):
        """Save ablation results to file"""
        results_path = self.results_dir / "ablation_results.json"
        with open(results_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)
    
    def _save_dose_response_curve(self, curve: DoseResponseCurve):
        """Save dose-response curve to file"""
        curve_path = self.results_dir / f"dose_response_{curve.effect_name}.json"
        with open(curve_path, 'w') as f:
            json.dump(asdict(curve), f, indent=2, default=str)
    
    def _save_interaction_analyses(self, analyses: List[InteractionAnalysis]):
        """Save interaction analyses to file"""
        analyses_path = self.results_dir / "interaction_analyses.json"
        with open(analyses_path, 'w') as f:
            json.dump([asdict(a) for a in analyses], f, indent=2, default=str)

def main():
    """Example usage of the ablations analyzer"""
    analyzer = AblationsAnalyzer()
    
    # Example pack configurations
    pack_configs = {
        "caffeine": {
            "name": "caffeine",
            "effects": [
                {"effect": "attention_enhancement", "weight": 1.0, "direction": "up"},
                {"effect": "alertness_boost", "weight": 0.8, "direction": "up"}
            ]
        },
        "lsd": {
            "name": "lsd", 
            "effects": [
                {"effect": "perceptual_distortion", "weight": 1.0, "direction": "up"},
                {"effect": "creativity_enhancement", "weight": 0.9, "direction": "up"}
            ]
        }
    }
    
    # Mock test data
    test_data = [{"prompt": f"Test prompt {i}", "expected": "response"} for i in range(100)]
    
    # Mock baseline results
    baseline_results = {
        "caffeine_score": 0.75,
        "lsd_score": 0.68
    }
    
    print("Running component ablations...")
    ablation_results = analyzer.run_component_ablations(
        pack_configs, baseline_results, test_data
    )
    
    print("Running dose-response analysis...")
    dose_response = analyzer.run_dose_response_analysis(
        "attention_enhancement", pack_configs["caffeine"], 
        [0.0, 0.25, 0.5, 0.75, 1.0], test_data
    )
    
    print("Analyzing effect interactions...")
    interactions = analyzer.analyze_effect_interactions(
        pack_configs, test_data
    )
    
    print("Generating report...")
    report = analyzer.generate_ablation_report()
    
    print(f"Analysis complete! Results saved to: {analyzer.results_dir}")
    print(f"Total ablations: {len(ablation_results)}")
    print(f"Dose-response curves: {len(analyzer.dose_response_curves)}")
    print(f"Interactions analyzed: {len(interactions)}")

if __name__ == "__main__":
    main()
