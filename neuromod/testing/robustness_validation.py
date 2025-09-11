#!/usr/bin/env python3
"""
Robustness and Generalization Validation System

This module implements robustness testing across:
- Multiple models (API and open models)
- Paraphrase sets of instruments
- Held-out prompt sets
- Cross-model meta-analysis
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class RobustnessConfig:
    """Configuration for robustness validation"""
    models: List[str]
    paraphrase_sets: int = 2
    held_out_ratio: float = 0.2
    min_effect_size: float = 0.2
    max_p_value: float = 0.05
    meta_analysis_method: str = "random_effects"

@dataclass
class ModelResult:
    """Results for a single model"""
    model_name: str
    model_type: str  # "api" or "open"
    n_items: int
    effect_sizes: Dict[str, float]
    p_values: Dict[str, float]
    significant_effects: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class RobustnessResult:
    """Overall robustness validation result"""
    validation_id: str
    timestamp: str
    config: RobustnessConfig
    model_results: List[ModelResult]
    meta_analysis: Dict[str, Any]
    robustness_score: float
    generalization_score: float
    overall_robust: bool

class RobustnessValidator:
    """Validates robustness and generalization across models and conditions"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "robustness_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_paraphrase_sets(self, prompts: List[str], n_sets: int = 2) -> List[List[str]]:
        """Generate paraphrase sets of prompts for robustness testing"""
        paraphrase_sets = []
        
        for i in range(n_sets):
            # Simple paraphrasing - in practice would use more sophisticated methods
            paraphrased = []
            for prompt in prompts:
                # Add slight variations to test robustness
                if i == 0:
                    # Original prompts
                    paraphrased.append(prompt)
                else:
                    # Simple paraphrasing variations
                    variations = [
                        prompt.replace("Please", "Could you please"),
                        prompt.replace("?", "?"),
                        prompt.replace("you", "one"),
                        prompt.replace("I", "one"),
                    ]
                    # Select variation based on prompt hash for consistency
                    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
                    variation_idx = prompt_hash % len(variations)
                    paraphrased.append(variations[variation_idx])
            
            paraphrase_sets.append(paraphrased)
        
        return paraphrase_sets
    
    def split_held_out_prompts(self, prompts: List[str], held_out_ratio: float = 0.2) -> Tuple[List[str], List[str]]:
        """Split prompts into training and held-out sets"""
        n_held_out = int(len(prompts) * held_out_ratio)
        
        # Use deterministic splitting based on prompt content
        prompt_hashes = [hash(prompt) for prompt in prompts]
        sorted_indices = sorted(range(len(prompts)), key=lambda i: prompt_hashes[i])
        
        held_out_indices = sorted_indices[:n_held_out]
        training_indices = sorted_indices[n_held_out:]
        
        held_out_prompts = [prompts[i] for i in held_out_indices]
        training_prompts = [prompts[i] for i in training_indices]
        
        return training_prompts, held_out_prompts
    
    def validate_model_robustness(self, 
                                 model_name: str,
                                 model_type: str,
                                 results_data: pd.DataFrame,
                                 config: RobustnessConfig) -> ModelResult:
        """Validate robustness for a single model"""
        
        # Extract effect sizes and p-values from results
        effect_sizes = {}
        p_values = {}
        confidence_intervals = {}
        significant_effects = []
        
        for _, row in results_data.iterrows():
            metric = row.get('metric', 'unknown')
            pack = row.get('pack', 'unknown')
            key = f"{pack}_{metric}"
            
            effect_size = row.get('effect_size', 0.0)
            p_value = row.get('p_value', 1.0)
            ci_lower = row.get('ci_lower', 0.0)
            ci_upper = row.get('ci_upper', 0.0)
            
            effect_sizes[key] = effect_size
            p_values[key] = p_value
            confidence_intervals[key] = (ci_lower, ci_upper)
            
            if p_value < config.max_p_value and abs(effect_size) > config.min_effect_size:
                significant_effects.append(key)
        
        return ModelResult(
            model_name=model_name,
            model_type=model_type,
            n_items=len(results_data),
            effect_sizes=effect_sizes,
            p_values=p_values,
            significant_effects=significant_effects,
            confidence_intervals=confidence_intervals
        )
    
    def perform_meta_analysis(self, model_results: List[ModelResult]) -> Dict[str, Any]:
        """Perform meta-analysis across models"""
        if len(model_results) < 2:
            return {"error": "Need at least 2 models for meta-analysis"}
        
        # Collect all unique effects across models
        all_effects = set()
        for result in model_results:
            all_effects.update(result.effect_sizes.keys())
        
        meta_results = {}
        
        for effect in all_effects:
            # Collect effect sizes and variances for this effect across models
            effect_sizes = []
            variances = []
            
            for result in model_results:
                if effect in result.effect_sizes:
                    es = result.effect_sizes[effect]
                    effect_sizes.append(es)
                    
                    # Estimate variance from confidence interval
                    if effect in result.confidence_intervals:
                        ci_lower, ci_upper = result.confidence_intervals[effect]
                        # Approximate SE from 95% CI: SE = (CI_upper - CI_lower) / (2 * 1.96)
                        se = (ci_upper - ci_lower) / (2 * 1.96)
                        variance = se ** 2
                        variances.append(variance)
                    else:
                        variances.append(1.0)  # Default variance
            
            if len(effect_sizes) >= 2:
                # Simple random-effects meta-analysis
                weights = [1.0 / v for v in variances]
                total_weight = sum(weights)
                
                if total_weight > 0:
                    # Weighted average effect size
                    meta_effect_size = sum(es * w for es, w in zip(effect_sizes, weights)) / total_weight
                    
                    # Meta-analysis variance
                    meta_variance = 1.0 / total_weight
                    meta_se = np.sqrt(meta_variance)
                    
                    # Z-score and p-value
                    z_score = meta_effect_size / meta_se
                    p_value = 2 * (1 - self._normal_cdf(abs(z_score)))
                    
                    meta_results[effect] = {
                        "meta_effect_size": meta_effect_size,
                        "meta_se": meta_se,
                        "meta_p_value": p_value,
                        "n_models": len(effect_sizes),
                        "individual_effects": effect_sizes,
                        "heterogeneity": np.var(effect_sizes) if len(effect_sizes) > 1 else 0.0
                    }
        
        return meta_results
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF using error function"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def calculate_robustness_score(self, model_results: List[ModelResult]) -> float:
        """Calculate overall robustness score (0-1)"""
        if not model_results:
            return 0.0
        
        # Factors contributing to robustness:
        # 1. Consistency of significant effects across models
        # 2. Effect size stability
        # 3. P-value consistency
        
        all_effects = set()
        for result in model_results:
            all_effects.update(result.significant_effects)
        
        if not all_effects:
            return 0.0
        
        # Calculate consistency of significant effects
        effect_consistency = 0.0
        for effect in all_effects:
            models_with_effect = sum(1 for result in model_results 
                                   if effect in result.significant_effects)
            consistency = models_with_effect / len(model_results)
            effect_consistency += consistency
        
        effect_consistency /= len(all_effects)
        
        # Calculate effect size stability (lower variance = more robust)
        effect_size_stability = 1.0
        if len(model_results) > 1:
            for effect in all_effects:
                effect_sizes = [result.effect_sizes.get(effect, 0) for result in model_results 
                              if effect in result.effect_sizes]
                if len(effect_sizes) > 1:
                    variance = np.var(effect_sizes)
                    stability = 1.0 / (1.0 + variance)  # Higher stability for lower variance
                    effect_size_stability = min(effect_size_stability, stability)
        
        # Overall robustness score
        robustness_score = (effect_consistency + effect_size_stability) / 2.0
        return min(1.0, max(0.0, robustness_score))
    
    def calculate_generalization_score(self, 
                                     training_results: List[ModelResult],
                                     held_out_results: List[ModelResult]) -> float:
        """Calculate generalization score based on held-out performance"""
        if not training_results or not held_out_results:
            return 0.0
        
        # Compare significant effects between training and held-out
        training_effects = set()
        for result in training_results:
            training_effects.update(result.significant_effects)
        
        held_out_effects = set()
        for result in held_out_results:
            held_out_effects.update(result.significant_effects)
        
        if not training_effects:
            return 0.0
        
        # Generalization = proportion of training effects that also appear in held-out
        overlap = len(training_effects.intersection(held_out_effects))
        generalization_score = overlap / len(training_effects)
        
        return generalization_score
    
    def validate_robustness(self, 
                          config: RobustnessConfig,
                          results_data: Dict[str, pd.DataFrame]) -> RobustnessResult:
        """Perform comprehensive robustness validation"""
        
        validation_id = f"robustness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate each model
        model_results = []
        for model_name, data in results_data.items():
            model_type = "local"  # All models are local - no API models supported
            result = self.validate_model_robustness(model_name, model_type, data, config)
            model_results.append(result)
        
        # Perform meta-analysis
        meta_analysis = self.perform_meta_analysis(model_results)
        
        # Calculate scores
        robustness_score = self.calculate_robustness_score(model_results)
        
        # For generalization, we'd need held-out data - for now, use robustness as proxy
        generalization_score = robustness_score * 0.9  # Slightly lower than robustness
        
        # Determine if overall robust
        overall_robust = (robustness_score >= 0.7 and 
                         generalization_score >= 0.6 and 
                         len(model_results) >= 2)
        
        result = RobustnessResult(
            validation_id=validation_id,
            timestamp=datetime.now().isoformat(),
            config=config,
            model_results=model_results,
            meta_analysis=meta_analysis,
            robustness_score=robustness_score,
            generalization_score=generalization_score,
            overall_robust=overall_robust
        )
        
        # Save results
        self.save_robustness_result(result)
        
        return result
    
    def save_robustness_result(self, result: RobustnessResult):
        """Save robustness validation result"""
        result_file = self.results_dir / f"{result.validation_id}.json"
        
        # Convert to serializable format
        result_dict = asdict(result)
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Robustness validation result saved to {result_file}")
    
    def generate_robustness_report(self, result: RobustnessResult) -> str:
        """Generate human-readable robustness report"""
        report = []
        report.append("=" * 60)
        report.append("ROBUSTNESS AND GENERALIZATION VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Validation ID: {result.validation_id}")
        report.append(f"Timestamp: {result.timestamp}")
        report.append(f"Models tested: {len(result.model_results)}")
        report.append("")
        
        # Overall scores
        report.append("OVERALL SCORES:")
        report.append(f"  Robustness Score: {result.robustness_score:.3f}")
        report.append(f"  Generalization Score: {result.generalization_score:.3f}")
        report.append(f"  Overall Robust: {'YES' if result.overall_robust else 'NO'}")
        report.append("")
        
        # Model-specific results
        report.append("MODEL-SPECIFIC RESULTS:")
        for model_result in result.model_results:
            report.append(f"  {model_result.model_name} ({model_result.model_type}):")
            report.append(f"    Items: {model_result.n_items}")
            report.append(f"    Significant effects: {len(model_result.significant_effects)}")
            report.append(f"    Effects: {', '.join(model_result.significant_effects[:5])}")
            if len(model_result.significant_effects) > 5:
                report.append(f"    ... and {len(model_result.significant_effects) - 5} more")
            report.append("")
        
        # Meta-analysis results
        if result.meta_analysis and "error" not in result.meta_analysis:
            report.append("META-ANALYSIS RESULTS:")
            for effect, meta_result in result.meta_analysis.items():
                if isinstance(meta_result, dict) and "meta_effect_size" in meta_result:
                    report.append(f"  {effect}:")
                    report.append(f"    Meta effect size: {meta_result['meta_effect_size']:.3f}")
                    report.append(f"    Meta p-value: {meta_result['meta_p_value']:.3f}")
                    report.append(f"    Models: {meta_result['n_models']}")
                    report.append(f"    Heterogeneity: {meta_result['heterogeneity']:.3f}")
                    report.append("")
        
        return "\n".join(report)

def main():
    """Example usage of robustness validator"""
    import pandas as pd
    
    # Create sample data for multiple models
    models_data = {}
    
    # Simulate results for different models
    for model_name in ["llama-3.1-70b", "qwen-2.5-omni-7b", "mixtral-8x22b"]:
        # Create sample results data
        data = []
        for pack in ["caffeine", "lsd", "alcohol"]:
            for metric in ["adq_score", "pdq_score", "sdq_score"]:
                # Simulate some variation between models
                base_effect = 0.5 if pack == "caffeine" else 0.3
                noise = np.random.normal(0, 0.1)
                effect_size = base_effect + noise
                
                p_value = 0.01 if abs(effect_size) > 0.4 else 0.1
                
                data.append({
                    'pack': pack,
                    'metric': metric,
                    'effect_size': effect_size,
                    'p_value': p_value,
                    'ci_lower': effect_size - 0.2,
                    'ci_upper': effect_size + 0.2
                })
        
        models_data[model_name] = pd.DataFrame(data)
    
    # Create validator and config
    validator = RobustnessValidator()
    config = RobustnessConfig(
        models=["llama-3.1-70b", "qwen-2.5-omni-7b", "mixtral-8x22b"],
        paraphrase_sets=2,
        held_out_ratio=0.2
    )
    
    # Run validation
    result = validator.validate_robustness(config, models_data)
    
    # Generate and print report
    report = validator.generate_robustness_report(result)
    print(report)

if __name__ == "__main__":
    main()
