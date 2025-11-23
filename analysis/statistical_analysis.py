#!/usr/bin/env python3
"""
Statistical Analysis System for Neuromodulation Study

This module implements comprehensive statistical analysis including:
- Paired t-tests and Wilcoxon signed-rank tests
- Benjamini-Hochberg FDR correction
- Effect size calculations (Cohen's d, Cliff's delta)
- Bootstrap confidence intervals
- Mixed-effects models
- Bayesian hierarchical models
- Canonical correlation analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# import statsmodels.api as sm
# from statsmodels.stats.multitest import multipletests
# from statsmodels.formula.api import mixedlm
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

# Import advanced statistics
try:
    from neuromod.testing.advanced_statistics import AdvancedStatisticalAnalyzer
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False
    AdvancedStatisticalAnalyzer = None

logger = logging.getLogger(__name__)

@dataclass
class StatisticalResult:
    """Result of a statistical test"""
    test_name: str
    metric: str
    pack_name: str
    n: int
    statistic: float
    p_value: float
    p_value_fdr: float
    effect_size: float
    effect_size_type: str
    ci_lower: float
    ci_upper: float
    significant: bool
    interpretation: str

class StatisticalAnalyzer:
    """Comprehensive statistical analysis for neuromodulation study"""
    
    def __init__(self, alpha: float = 0.05, n_bootstrap: int = 10000):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        
        # Initialize advanced statistics analyzer if available
        if ADVANCED_STATS_AVAILABLE:
            self.advanced_stats = AdvancedStatisticalAnalyzer()
        else:
            self.advanced_stats = None
            logger.warning("AdvancedStatisticalAnalyzer not available. Mixed-effects models will not work.")
        
    def paired_t_test(self, control: np.ndarray, treatment: np.ndarray, 
                     metric_name: str, pack_name: str) -> StatisticalResult:
        """Perform paired t-test"""
        if len(control) != len(treatment):
            raise ValueError("Control and treatment arrays must have same length")
        
        # Remove NaN values
        mask = ~(np.isnan(control) | np.isnan(treatment))
        control_clean = control[mask]
        treatment_clean = treatment[mask]
        
        if len(control_clean) < 2:
            return StatisticalResult(
                test_name="paired_t_test",
                metric=metric_name,
                pack_name=pack_name,
                n=0,
                statistic=np.nan,
                p_value=np.nan,
                p_value_fdr=np.nan,
                effect_size=np.nan,
                effect_size_type="cohens_d",
                ci_lower=np.nan,
                ci_upper=np.nan,
                significant=False,
                interpretation="Insufficient data"
            )
        
        # Perform t-test
        statistic, p_value = ttest_rel(treatment_clean, control_clean)
        
        # Calculate Cohen's d
        diff = treatment_clean - control_clean
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(control_clean, treatment_clean)
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Interpretation
        interpretation = self._interpret_effect_size(cohens_d, significant)
        
        return StatisticalResult(
            test_name="paired_t_test",
            metric=metric_name,
            pack_name=pack_name,
            n=len(control_clean),
            statistic=statistic,
            p_value=p_value,
            p_value_fdr=np.nan,  # Will be filled by FDR correction
            effect_size=cohens_d,
            effect_size_type="cohens_d",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=significant,
            interpretation=interpretation
        )
    
    def wilcoxon_test(self, control: np.ndarray, treatment: np.ndarray,
                     metric_name: str, pack_name: str) -> StatisticalResult:
        """Perform Wilcoxon signed-rank test"""
        if len(control) != len(treatment):
            raise ValueError("Control and treatment arrays must have same length")
        
        # Remove NaN values
        mask = ~(np.isnan(control) | np.isnan(treatment))
        control_clean = control[mask]
        treatment_clean = treatment[mask]
        
        if len(control_clean) < 2:
            return StatisticalResult(
                test_name="wilcoxon_test",
                metric=metric_name,
                pack_name=pack_name,
                n=0,
                statistic=np.nan,
                p_value=np.nan,
                p_value_fdr=np.nan,
                effect_size=np.nan,
                effect_size_type="cliffs_delta",
                ci_lower=np.nan,
                ci_upper=np.nan,
                significant=False,
                interpretation="Insufficient data"
            )
        
        # Perform Wilcoxon test
        statistic, p_value = wilcoxon(treatment_clean, control_clean, alternative='two-sided')
        
        # Calculate Cliff's delta
        cliffs_delta = self._calculate_cliffs_delta(control_clean, treatment_clean)
        
        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ci(control_clean, treatment_clean, method='wilcoxon')
        
        # Determine significance
        significant = p_value < self.alpha
        
        # Interpretation
        interpretation = self._interpret_effect_size(cliffs_delta, significant)
        
        return StatisticalResult(
            test_name="wilcoxon_test",
            metric=metric_name,
            pack_name=pack_name,
            n=len(control_clean),
            statistic=statistic,
            p_value=p_value,
            p_value_fdr=np.nan,  # Will be filled by FDR correction
            effect_size=cliffs_delta,
            effect_size_type="cliffs_delta",
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significant=significant,
            interpretation=interpretation
        )
    
    def _calculate_cliffs_delta(self, control: np.ndarray, treatment: np.ndarray) -> float:
        """Calculate Cliff's delta effect size"""
        n_control = len(control)
        n_treatment = len(treatment)
        
        # Count pairs where treatment > control
        greater_count = 0
        for t_val in treatment:
            for c_val in control:
                if t_val > c_val:
                    greater_count += 1
        
        # Count pairs where treatment < control
        less_count = 0
        for t_val in treatment:
            for c_val in control:
                if t_val < c_val:
                    less_count += 1
        
        # Calculate Cliff's delta
        total_pairs = n_control * n_treatment
        cliffs_delta = (greater_count - less_count) / total_pairs
        
        return cliffs_delta
    
    def _bootstrap_ci(self, control: np.ndarray, treatment: np.ndarray, 
                     method: str = 't_test', confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        if len(control) < 3 or len(treatment) < 3:
            return np.nan, np.nan
        
        bootstrap_stats = []
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            control_boot = np.random.choice(control, size=len(control), replace=True)
            treatment_boot = np.random.choice(treatment, size=len(treatment), replace=True)
            
            if method == 't_test':
                # Bootstrap t-test statistic
                diff = treatment_boot - control_boot
                if np.std(diff, ddof=1) > 0:
                    stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff)))
                    bootstrap_stats.append(stat)
            else:
                # Bootstrap Wilcoxon statistic
                try:
                    stat, _ = wilcoxon(treatment_boot, control_boot, alternative='two-sided')
                    bootstrap_stats.append(stat)
                except:
                    continue
        
        if not bootstrap_stats:
            return np.nan, np.nan
        
        # Calculate confidence interval
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _interpret_effect_size(self, effect_size: float, significant: bool) -> str:
        """Interpret effect size magnitude"""
        abs_effect = abs(effect_size)
        
        if abs_effect < 0.2:
            magnitude = "negligible"
        elif abs_effect < 0.5:
            magnitude = "small"
        elif abs_effect < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        significance = "significant" if significant else "not significant"
        
        return f"{magnitude} effect size ({effect_size:.3f}), {significance}"
    
    def apply_fdr_correction(self, results: List[StatisticalResult]) -> List[StatisticalResult]:
        """Apply Benjamini-Hochberg FDR correction"""
        if not results:
            return results
        
        # Extract p-values and track their original indices
        p_values = []
        valid_indices = []  # Track which results have valid p-values
        
        for idx, result in enumerate(results):
            if not np.isnan(result.p_value):
                p_values.append(result.p_value)
                valid_indices.append(idx)
        
        if not p_values:
            return results
        
        # Benjamini-Hochberg FDR correction
        n = len(p_values)
        p_array = np.array(p_values)
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        # Calculate FDR corrected p-values
        p_corrected = np.zeros(n)
        for i in range(n):
            p_corrected[i] = sorted_p_values[i] * n / (i + 1)
        
        # Ensure monotonicity (BH step-up procedure)
        for i in range(n-2, -1, -1):
            p_corrected[i] = min(p_corrected[i], p_corrected[i+1])
        
        # Cap at 1.0
        p_corrected = np.minimum(p_corrected, 1.0)
        
        # Map corrected values back to original order
        p_corrected_original_order = np.zeros(n)
        for sorted_idx, original_idx in enumerate(sorted_indices):
            p_corrected_original_order[original_idx] = p_corrected[sorted_idx]
        
        # Determine which tests are rejected
        rejected = p_corrected_original_order < self.alpha
        
        # Update results with corrected p-values
        corrected_results = []
        valid_idx = 0
        
        for idx, result in enumerate(results):
            if np.isnan(result.p_value):
                # Keep NaN results unchanged
                corrected_results.append(result)
            else:
                # Map back to the correct corrected p-value
                result.p_value_fdr = p_corrected_original_order[valid_idx]
                result.significant = rejected[valid_idx]
                corrected_results.append(result)
                valid_idx += 1
        
        return corrected_results
    
    def mixed_effects_model(self, data: pd.DataFrame, formula: str, 
                           group_var: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform Mixed-Effects Analysis using statsmodels (via AdvancedStatisticalAnalyzer).
        
        Args:
            data: DataFrame with the data
            formula: Model formula (e.g., "score ~ condition")
            group_var: Grouping variable for random effects (e.g., "prompt_id", "item_id")
                     If None, will try to auto-detect from common column names
        
        Returns:
            Dictionary with model results including:
            - success: bool
            - p_values: Dict of p-values for fixed effects
            - coefficients: Dict of coefficients for fixed effects
            - aic: AIC value
            - bic: BIC value
            - summary: Model summary string
            - error: Error message if failed
        """
        if self.advanced_stats is None:
            return {
                'success': False,
                'error': 'AdvancedStatisticalAnalyzer not available. Please install statsmodels.',
                'model': None
            }
        
        try:
            # Auto-detect group_var if not provided
            if group_var is None:
                # Try common grouping variable names
                possible_group_vars = ['prompt_id', 'item_id', 'seed', 'trial_id', 'replicate_id', 'group']
                for var in possible_group_vars:
                    if var in data.columns:
                        group_var = var
                        logger.info(f"Auto-detected group_var: {group_var}")
                        break
                
                if group_var is None:
                    # Try to find any categorical column that might be a grouping variable
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                    if categorical_cols:
                        # Prefer columns with fewer unique values (more likely to be grouping vars)
                        for col in sorted(categorical_cols, key=lambda x: data[x].nunique()):
                            if data[col].nunique() < len(data) / 2:  # Reasonable grouping variable
                                group_var = col
                                logger.info(f"Auto-detected group_var from categorical columns: {group_var}")
                                break
                
                if group_var is None:
                    raise ValueError("Could not auto-detect group_var. Please specify it explicitly.")
            
            # Ensure group_var exists in data
            if group_var not in data.columns:
                raise ValueError(f"group_var '{group_var}' not found in data columns: {list(data.columns)}")
            
            # Fit the mixed-effects model
            result = self.advanced_stats.fit_mixed_effects_model(
                data=data,
                formula=formula,
                group_var=group_var
            )
            
            return {
                'success': True,
                'p_values': result.fixed_effects_pvalues,
                'coefficients': result.fixed_effects,
                'coefficients_se': result.fixed_effects_se,
                'coefficients_ci': result.fixed_effects_ci,
                'random_effects': result.random_effects,
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.log_likelihood,
                'n_observations': result.n_observations,
                'n_groups': result.n_groups,
                'convergence_warning': result.convergence_warning,
                'summary': result.model_summary,
                'formula': result.formula,
                'model_name': result.model_name
            }
            
        except Exception as e:
            logger.error(f"Failed to fit mixed-effects model: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'model': None
            }
    
    def canonical_correlation_analysis(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Perform canonical correlation analysis"""
        try:
            from sklearn.cross_decomposition import CCA
            
            # Standardize data
            scaler_X = StandardScaler()
            scaler_Y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)
            Y_scaled = scaler_Y.fit_transform(Y)
            
            # Fit CCA
            cca = CCA(n_components=min(X.shape[1], Y.shape[1]))
            X_c, Y_c = cca.fit_transform(X_scaled, Y_scaled)
            
            # Calculate canonical correlations
            correlations = []
            for i in range(X_c.shape[1]):
                corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                correlations.append(corr)
            
            return {
                'success': True,
                'canonical_correlations': correlations,
                'n_components': len(correlations),
                'X_canonical': X_c,
                'Y_canonical': Y_c,
                'X_loadings': cca.x_loadings_,
                'Y_loadings': cca.y_loadings_
            }
        except Exception as e:
            logger.error(f"Canonical correlation analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def roc_analysis(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """Perform ROC analysis"""
        try:
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            
            # Find optimal threshold (Youden's J statistic)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            return {
                'success': True,
                'auc': auc,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'optimal_threshold': optimal_threshold,
                'optimal_tpr': tpr[optimal_idx],
                'optimal_fpr': fpr[optimal_idx]
            }
        except Exception as e:
            logger.error(f"ROC analysis failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_experiment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive analysis of experiment data"""
        results = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'n_observations': len(data),
            'metrics_analyzed': [],
            'packs_analyzed': [],
            'statistical_tests': [],
            'fdr_correction': {},
            'summary': {}
        }
        
        # Get unique metrics and packs
        metrics = data['metric'].unique() if 'metric' in data.columns else []
        packs = data['pack'].unique() if 'pack' in data.columns else []
        
        results['metrics_analyzed'] = list(metrics)
        results['packs_analyzed'] = list(packs)
        
        # Perform statistical tests for each metric-pack combination
        all_test_results = []
        
        for metric in metrics:
            for pack in packs:
                if pack == 'control':
                    continue
                
                # Get data for this metric and pack
                pack_data = data[(data['pack'] == pack) & (data['metric'] == metric)]
                control_data = data[(data['pack'] == 'control') & (data['metric'] == metric)]
                
                if len(pack_data) == 0 or len(control_data) == 0:
                    continue
                
                # Ensure we have paired data
                if 'item_id' in data.columns:
                    # Merge on item_id to get paired data
                    merged = pd.merge(
                        pack_data[['item_id', 'score']].rename(columns={'score': 'treatment'}),
                        control_data[['item_id', 'score']].rename(columns={'score': 'control'}),
                        on='item_id'
                    )
                    
                    if len(merged) == 0:
                        continue
                    
                    control_scores = merged['control'].values
                    treatment_scores = merged['treatment'].values
                else:
                    # Use all available data (not ideal for paired tests)
                    control_scores = control_data['score'].values
                    treatment_scores = pack_data['score'].values
                
                # Perform paired t-test
                t_test_result = self.paired_t_test(control_scores, treatment_scores, metric, pack)
                all_test_results.append(t_test_result)
                
                # Perform Wilcoxon test
                wilcoxon_result = self.wilcoxon_test(control_scores, treatment_scores, metric, pack)
                all_test_results.append(wilcoxon_result)
        
        # Apply FDR correction
        corrected_results = self.apply_fdr_correction(all_test_results)
        results['statistical_tests'] = [asdict(r) for r in corrected_results]
        
        # Calculate FDR correction info
        valid_p_values = [r.p_value for r in corrected_results if not np.isnan(r.p_value)]
        significant_raw = [r for r in all_test_results if r.p_value < self.alpha and not np.isnan(r.p_value)]
        significant_fdr = [r for r in corrected_results if r.significant]
        
        results['fdr_correction'] = {
            'n_tests': len(valid_p_values),
            'n_significant_raw': len(significant_raw),
            'n_significant_fdr': len(significant_fdr),
            'fdr_threshold': self.alpha,
            'method': 'Benjamini-Hochberg'
        }
        
        # Summary statistics
        significant_tests = [r for r in corrected_results if r.significant]
        results['summary'] = {
            'total_tests': len(corrected_results),
            'significant_tests': len(significant_tests),
            'significant_rate': len(significant_tests) / len(corrected_results) if corrected_results else 0,
            'effect_sizes': {
                'mean_cohens_d': np.mean([r.effect_size for r in corrected_results if r.effect_size_type == 'cohens_d' and not np.isnan(r.effect_size)]) if any(r.effect_size_type == 'cohens_d' and not np.isnan(r.effect_size) for r in corrected_results) else np.nan,
                'mean_cliffs_delta': np.mean([r.effect_size for r in corrected_results if r.effect_size_type == 'cliffs_delta' and not np.isnan(r.effect_size)]) if any(r.effect_size_type == 'cliffs_delta' and not np.isnan(r.effect_size) for r in corrected_results) else np.nan
            }
        }
        
        return results

def main():
    """Example usage of the statistical analyzer"""
    # Create sample data
    np.random.seed(42)
    n_items = 50
    
    data = []
    for item_id in range(n_items):
        # Control condition
        control_score = np.random.normal(0, 1)
        data.append({'item_id': item_id, 'pack': 'control', 'metric': 'adq_score', 'score': control_score})
        
        # Treatment condition (with effect)
        treatment_score = control_score + np.random.normal(0.5, 0.5)  # 0.5 effect size
        data.append({'item_id': item_id, 'pack': 'caffeine', 'metric': 'adq_score', 'score': treatment_score})
    
    df = pd.DataFrame(data)
    
    # Create analyzer
    analyzer = StatisticalAnalyzer()
    
    # Analyze experiment
    results = analyzer.analyze_experiment(df)
    
    print("Statistical Analysis Results:")
    print(f"Total tests: {results['summary']['total_tests']}")
    print(f"Significant tests: {results['summary']['significant_tests']}")
    print(f"Significant rate: {results['summary']['significant_rate']:.3f}")
    print(f"Mean Cohen's d: {results['summary']['effect_sizes']['mean_cohens_d']:.3f}")
    
    # Show individual test results
    print("\nIndividual Test Results:")
    for test in results['statistical_tests']:
        if test['significant']:
            print(f"  {test['pack_name']} - {test['metric']}: {test['test_name']} p={test['p_value_fdr']:.3f}, d={test['effect_size']:.3f}")

if __name__ == "__main__":
    main()
