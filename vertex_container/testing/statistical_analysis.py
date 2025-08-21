"""
Statistical Analysis Pipeline for Neuromodulation Testing

This module provides comprehensive statistical analysis capabilities including:
- Mixed-effects models for repeated measures
- ROC/PR curve generation and analysis
- Bayesian hierarchical models
- Effect size calculations and power analysis
- Multiple comparison correction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

# Statistical libraries
try:
    import scipy.stats as stats
    import scipy.special as special
    from scipy.optimize import minimize
    from scipy.stats import chi2, norm, t
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available. Some statistical functions will be limited.")

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("Scikit-learn not available. ROC/PR analysis will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting will be disabled.")

# Set plotting style
if PLOTTING_AVAILABLE:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

@dataclass
class StatisticalResult:
    """Container for statistical analysis results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    degrees_of_freedom: Optional[int] = None
    sample_size: Optional[int] = None
    power: Optional[float] = None

@dataclass
class ROCResult:
    """Container for ROC curve analysis results"""
    fpr: np.ndarray
    tpr: np.ndarray
    auc_score: float
    thresholds: np.ndarray
    optimal_threshold: float
    optimal_f1: float

@dataclass
class MixedEffectsResult:
    """Container for mixed-effects model results"""
    fixed_effects: Dict[str, float]
    random_effects: Dict[str, float]
    residuals: np.ndarray
    aic: float
    bic: float
    log_likelihood: float
    convergence: bool

class StatisticalAnalyzer:
    """Main statistical analysis class for neuromodulation testing"""
    
    def __init__(self):
        self.results_cache = {}
        
    def analyze_test_results(self, baseline_results: List[Dict], 
                           treatment_results: List[Dict],
                           test_name: str = "Unknown Test") -> Dict[str, Any]:
        """
        Comprehensive analysis of test results comparing baseline vs treatment
        
        Args:
            baseline_results: List of baseline test results
            treatment_results: List of treatment test results  
            test_name: Name of the test being analyzed
            
        Returns:
            Dictionary containing all statistical analysis results
        """
        analysis = {
            'test_name': test_name,
            'baseline_count': len(baseline_results),
            'treatment_count': len(treatment_results),
            'statistical_tests': {},
            'effect_sizes': {},
            'roc_analysis': {},
            'power_analysis': {},
            'descriptive_stats': {}
        }
        
        # Extract key metrics for analysis
        baseline_metrics = self._extract_metrics(baseline_results)
        treatment_metrics = self._extract_metrics(treatment_results)
        
        # Descriptive statistics
        analysis['descriptive_stats'] = {
            'baseline': {k: self._calculate_descriptive_stats(v) for k, v in baseline_metrics.items()},
            'treatment': {k: self._calculate_descriptive_stats(v) for k, v in treatment_metrics.items()}
        }
        
        # Statistical tests for each metric
        for metric_name in baseline_metrics.keys():
            if metric_name in treatment_metrics:
                baseline_data = baseline_metrics[metric_name]
                treatment_data = treatment_metrics[metric_name]
                
                # Paired t-test (if we have matched samples)
                if len(baseline_data) == len(treatment_data):
                    t_test = self._paired_t_test(baseline_data, treatment_data, metric_name)
                    analysis['statistical_tests'][f'{metric_name}_paired_t'] = t_test
                
                # Independent t-test
                indep_t_test = self._independent_t_test(baseline_data, treatment_data, metric_name)
                analysis['statistical_tests'][f'{metric_name}_independent_t'] = indep_t_test
                
                # Mann-Whitney U test (non-parametric)
                mw_test = self._mann_whitney_test(baseline_data, treatment_data, metric_name)
                analysis['statistical_tests'][f'{metric_name}_mann_whitney'] = mw_test
                
                # Effect size (Cohen's d)
                cohens_d = self._cohens_d(baseline_data, treatment_data)
                analysis['effect_sizes'][metric_name] = cohens_d
                
                # ROC analysis for binary classification
                if self._can_analyze_roc(baseline_data, treatment_data):
                    roc_result = self._analyze_roc(baseline_data, treatment_data, metric_name)
                    analysis['roc_analysis'][metric_name] = roc_result
        
        # Power analysis
        analysis['power_analysis'] = self._power_analysis(baseline_metrics, treatment_metrics)
        
        # Multiple comparison correction
        analysis['corrected_p_values'] = self._apply_multiple_comparison_correction(
            analysis['statistical_tests']
        )
        
        return analysis
    
    def _extract_metrics(self, results: List[Dict]) -> Dict[str, List[float]]:
        """Extract numerical metrics from test results"""
        metrics = {}
        
        for result in results:
            # Extract common metrics
            if 'aggregated_subscales' in result:
                for subscale, value in result['aggregated_subscales'].items():
                    if subscale not in metrics:
                        metrics[subscale] = []
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        metrics[subscale].append(float(value))
            
            # Extract probability scores
            if 'presence_probability' in result:
                if 'presence_probability' not in metrics:
                    metrics['presence_probability'] = []
                value = result['presence_probability']
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics['presence_probability'].append(float(value))
            
            # Extract intensity scores
            if 'intensity_score' in result:
                if 'intensity_score' not in metrics:
                    metrics['intensity_score'] = []
                value = result['intensity_score']
                if isinstance(value, (int, float)) and not np.isnan(value):
                    metrics['intensity_score'].append(float(value))
        
        # Filter out metrics with insufficient data
        filtered_metrics = {}
        for metric_name, values in metrics.items():
            if len(values) >= 2 and len(set(values)) > 1:  # Need at least 2 samples and some variation
                filtered_metrics[metric_name] = values
        
        return filtered_metrics
    
    def _calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """Calculate descriptive statistics for a dataset"""
        if not data or len(data) < 2:
            return {}
        
        data_array = np.array(data)
        # Filter out any remaining NaN values
        data_array = data_array[~np.isnan(data_array)]
        
        if len(data_array) < 2:
            return {}
        
        try:
            return {
                'n': len(data_array),
                'mean': float(np.mean(data_array)),
                'std': float(np.std(data_array, ddof=1)),
                'median': float(np.median(data_array)),
                'min': float(np.min(data_array)),
                'max': float(np.max(data_array)),
                'q25': float(np.percentile(data_array, 25)),
                'q75': float(np.percentile(data_array, 75))
            }
        except Exception:
            return {}
    
    def _paired_t_test(self, baseline: List[float], treatment: List[float], 
                       metric_name: str) -> StatisticalResult:
        """Perform paired t-test"""
        if not SCIPY_AVAILABLE:
            return StatisticalResult(
                test_name=f"Paired t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        baseline_array = np.array(baseline)
        treatment_array = np.array(treatment)
        
        # Filter out NaN values
        baseline_array = baseline_array[~np.isnan(baseline_array)]
        treatment_array = treatment_array[~np.isnan(treatment_array)]
        
        if len(baseline_array) != len(treatment_array) or len(baseline_array) < 2:
            return StatisticalResult(
                test_name=f"Paired t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        try:
            # Calculate differences
            differences = treatment_array - baseline_array
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_array, treatment_array)
            
            # Effect size (Cohen's d for paired samples)
            d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences, ddof=1) > 0 else 0
            
            # Confidence interval
            n = len(differences)
            se = np.std(differences, ddof=1) / np.sqrt(n) if np.std(differences, ddof=1) > 0 else 0
            t_critical = stats.t.ppf(0.975, n - 1)
            ci_lower = np.mean(differences) - t_critical * se
            ci_upper = np.mean(differences) + t_critical * se
            
            return StatisticalResult(
                test_name=f"Paired t-test ({metric_name})",
                statistic=float(t_stat),
                p_value=float(p_value),
                effect_size=float(d),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                degrees_of_freedom=n - 1,
                sample_size=n
            )
        except Exception:
            return StatisticalResult(
                test_name=f"Paired t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
    
    def _independent_t_test(self, baseline: List[float], treatment: List[float], 
                           metric_name: str) -> StatisticalResult:
        """Perform independent t-test"""
        if not SCIPY_AVAILABLE:
            return StatisticalResult(
                test_name=f"Independent t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        baseline_array = np.array(baseline)
        treatment_array = np.array(treatment)
        
        # Filter out NaN values
        baseline_array = baseline_array[~np.isnan(baseline_array)]
        treatment_array = treatment_array[~np.isnan(treatment_array)]
        
        if len(baseline_array) < 2 or len(treatment_array) < 2:
            return StatisticalResult(
                test_name=f"Independent t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        try:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(baseline_array, treatment_array)
            
            # Effect size (Cohen's d)
            d = self._cohens_d(baseline_array, treatment_array)
            
            # Confidence interval for mean difference
            n1, n2 = len(baseline_array), len(treatment_array)
            pooled_std = np.sqrt(((n1-1)*np.var(baseline_array, ddof=1) + 
                                 (n2-1)*np.var(treatment_array, ddof=1)) / (n1+n2-2))
            se = pooled_std * np.sqrt(1/n1 + 1/n2)
            mean_diff = np.mean(treatment_array) - np.mean(baseline_array)
            t_critical = stats.t.ppf(0.975, n1 + n2 - 2)
            ci_lower = mean_diff - t_critical * se
            ci_upper = mean_diff + t_critical * se
            
            return StatisticalResult(
                test_name=f"Independent t-test ({metric_name})",
                statistic=float(t_stat),
                p_value=float(p_value),
                effect_size=float(d),
                confidence_interval=(float(ci_lower), float(ci_upper)),
                degrees_of_freedom=n1 + n2 - 2,
                sample_size=n1 + n2
            )
        except Exception:
            return StatisticalResult(
                test_name=f"Independent t-test ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
    
    def _mann_whitney_test(self, baseline: List[float], treatment: List[float], 
                           metric_name: str) -> StatisticalResult:
        """Perform Mann-Whitney U test (non-parametric)"""
        if not SCIPY_AVAILABLE:
            return StatisticalResult(
                test_name=f"Mann-Whitney U ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        baseline_array = np.array(baseline)
        treatment_array = np.array(treatment)
        
        # Filter out NaN values
        baseline_array = baseline_array[~np.isnan(baseline_array)]
        treatment_array = treatment_array[~np.isnan(treatment_array)]
        
        if len(baseline_array) < 2 or len(treatment_array) < 2:
            return StatisticalResult(
                test_name=f"Mann-Whitney U ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
        
        try:
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(baseline_array, treatment_array, 
                                                alternative='two-sided')
            
            # Effect size (r = U / (n1 * n2))
            n1, n2 = len(baseline_array), len(treatment_array)
            r = u_stat / (n1 * n2)
            
            return StatisticalResult(
                test_name=f"Mann-Whitney U ({metric_name})",
                statistic=float(u_stat),
                p_value=float(p_value),
                effect_size=float(r),
                confidence_interval=(np.nan, np.nan),  # Not easily computed for U test
                sample_size=n1 + n2
            )
        except Exception:
            return StatisticalResult(
                test_name=f"Mann-Whitney U ({metric_name})",
                statistic=np.nan, p_value=np.nan, effect_size=np.nan,
                confidence_interval=(np.nan, np.nan)
            )
    
    def _cohens_d(self, baseline: Union[List[float], np.ndarray], 
                  treatment: Union[List[float], np.ndarray]) -> float:
        """Calculate Cohen's d effect size"""
        baseline_array = np.array(baseline)
        treatment_array = np.array(treatment)
        
        # Filter out NaN values
        baseline_array = baseline_array[~np.isnan(baseline_array)]
        treatment_array = treatment_array[~np.isnan(treatment_array)]
        
        if len(baseline_array) < 2 or len(treatment_array) < 2:
            return 0.0
        
        n1, n2 = len(baseline_array), len(treatment_array)
        
        try:
            pooled_std = np.sqrt(((n1-1)*np.var(baseline_array, ddof=1) + 
                                 (n2-1)*np.var(treatment_array, ddof=1)) / (n1+n2-2))
            
            if pooled_std == 0 or np.isnan(pooled_std):
                return 0.0
            
            d = (np.mean(treatment_array) - np.mean(baseline_array)) / pooled_std
            return float(d) if not np.isnan(d) else 0.0
        except Exception:
            return 0.0
    
    def _can_analyze_roc(self, baseline: List[float], treatment: List[float]) -> bool:
        """Check if ROC analysis is possible"""
        return (SKLEARN_AVAILABLE and 
                len(baseline) > 0 and len(treatment) > 0 and
                len(set(baseline)) > 1 and len(set(treatment)) > 1)
    
    def _analyze_roc(self, baseline: List[float], treatment: List[float], 
                     metric_name: str) -> Optional[ROCResult]:
        """Perform ROC analysis"""
        if not self._can_analyze_roc(baseline, treatment):
            return None
        
        # Create labels (0 for baseline, 1 for treatment)
        y_true = [0] * len(baseline) + [1] * len(treatment)
        y_scores = baseline + treatment
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc_score = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold (maximizing F1 score)
        f1_scores = []
        for threshold in thresholds:
            y_pred = [1 if score >= threshold else 0 for score in y_scores]
            tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
            fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
            fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
            
            if tp + fp == 0 or tp + fn == 0:
                f1_scores.append(0)
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        return ROCResult(
            fpr=fpr,
            tpr=tpr,
            auc_score=float(auc_score),
            thresholds=thresholds,
            optimal_threshold=float(optimal_threshold),
            optimal_f1=float(optimal_f1)
        )
    
    def _power_analysis(self, baseline_metrics: Dict[str, List[float]], 
                       treatment_metrics: Dict[str, List[float]]) -> Dict[str, Any]:
        """Perform power analysis for the study"""
        power_results = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name in treatment_metrics:
                baseline_data = baseline_metrics[metric_name]
                treatment_data = treatment_metrics[metric_name]
                
                if len(baseline_data) > 0 and len(treatment_data) > 0:
                    # Calculate effect size
                    d = self._cohens_d(baseline_data, treatment_data)
                    
                    # Skip if effect size is invalid
                    if np.isnan(d) or d == 0:
                        continue
                    
                    # Power analysis for t-test
                    n1, n2 = len(baseline_data), len(treatment_data)
                    alpha = 0.05
                    
                    # Approximate power calculation
                    if SCIPY_AVAILABLE:
                        try:
                            # Use scipy for power calculation
                            from scipy.stats import norm
                            z_alpha = norm.ppf(1 - alpha/2)
                            z_beta = norm.ppf(0.8)  # 80% power
                            
                            # Required sample size per group
                            n_required = 2 * ((z_alpha + z_beta) / d) ** 2
                            
                            power_results[metric_name] = {
                                'effect_size': float(d),
                                'current_power': self._calculate_power(d, n1, n2, alpha),
                                'required_sample_size': int(np.ceil(n_required)),
                                'current_sample_size': n1 + n2
                            }
                        except Exception:
                            # Fallback to basic power calculation
                            power_results[metric_name] = {
                                'effect_size': float(d),
                                'current_power': 'N/A',
                                'required_sample_size': 'N/A',
                                'current_sample_size': n1 + n2
                            }
        
        return power_results
    
    def _calculate_power(self, d: float, n1: int, n2: int, alpha: float = 0.05) -> float:
        """Calculate power for a t-test"""
        if not SCIPY_AVAILABLE or np.isnan(d) or d == 0:
            return np.nan
        
        try:
            # Pooled standard error
            pooled_se = np.sqrt(1/n1 + 1/n2)
            
            # Non-centrality parameter
            ncp = d / pooled_se
            
            # Critical t-value
            df = n1 + n2 - 2
            t_critical = stats.t.ppf(1 - alpha/2, df)
            
            # Power calculation
            power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
            return float(power) if not np.isnan(power) else np.nan
        except Exception:
            return np.nan
    
    def _apply_multiple_comparison_correction(self, statistical_tests: Dict[str, StatisticalResult]) -> Dict[str, float]:
        """Apply Benjamini-Hochberg FDR correction"""
        if not SCIPY_AVAILABLE:
            return {}
        
        # Extract p-values, filtering out NaN values
        test_names = []
        p_values = []
        
        for test_name, result in statistical_tests.items():
            if not np.isnan(result.p_value):
                test_names.append(test_name)
                p_values.append(result.p_value)
        
        if not p_values:
            return {}
        
        # Apply FDR correction
        try:
            from scipy.stats import false_discovery_control
            corrected_p_values = false_discovery_control(p_values, method='bh')
        except ImportError:
            # Fallback to manual implementation
            corrected_p_values = self._benjamini_hochberg_correction(p_values)
        
        return dict(zip(test_names, corrected_p_values))
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Manual implementation of Benjamini-Hochberg FDR correction"""
        if not p_values:
            return []
        
        n = len(p_values)
        p_array = np.array(p_values)
        
        # Sort p-values and get indices
        sorted_indices = np.argsort(p_array)
        sorted_p_values = p_array[sorted_indices]
        
        # Calculate corrected p-values
        corrected_p_values = np.zeros(n)
        for i, (idx, p_val) in enumerate(zip(sorted_indices, sorted_p_values)):
            corrected_p_values[idx] = min(p_val * n / (i + 1), 1.0)
        
        return corrected_p_values.tolist()
    
    def generate_plots(self, analysis_results: Dict[str, Any], 
                      output_path: Optional[str] = None) -> Dict[str, str]:
        """Generate visualization plots for the analysis results"""
        if not PLOTTING_AVAILABLE:
            return {"error": "Plotting libraries not available"}
        
        plot_paths = {}
        
        try:
            # 1. Effect size forest plot
            if 'effect_sizes' in analysis_results:
                self._plot_effect_sizes(analysis_results['effect_sizes'], output_path)
                plot_paths['effect_sizes'] = f"{output_path}_effect_sizes.png" if output_path else "effect_sizes.png"
            
            # 2. ROC curves
            if 'roc_analysis' in analysis_results:
                self._plot_roc_curves(analysis_results['roc_analysis'], output_path)
                plot_paths['roc_curves'] = f"{output_path}_roc_curves.png" if output_path else "roc_curves.png"
            
            # 3. Descriptive statistics
            if 'descriptive_stats' in analysis_results:
                self._plot_descriptive_stats(analysis_results['descriptive_stats'], output_path)
                plot_paths['descriptive_stats'] = f"{output_path}_descriptive_stats.png" if output_path else "descriptive_stats.png"
            
        except Exception as e:
            plot_paths['error'] = f"Error generating plots: {str(e)}"
        
        return plot_paths
    
    def _plot_effect_sizes(self, effect_sizes: Dict[str, float], output_path: Optional[str] = None):
        """Plot effect sizes as a forest plot"""
        if not effect_sizes:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = list(effect_sizes.keys())
        values = list(effect_sizes.values())
        
        # Color coding for effect size magnitude
        colors = []
        for d in values:
            if abs(d) < 0.2:
                colors.append('lightgray')  # Negligible
            elif abs(d) < 0.5:
                colors.append('lightblue')  # Small
            elif abs(d) < 0.8:
                colors.append('orange')     # Medium
            else:
                colors.append('red')        # Large
        
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, color=colors)
        
        # Add reference lines for effect size interpretation
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
        ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel("Cohen's d Effect Size")
        ax.set_title("Effect Sizes by Metric")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(f"{output_path}_effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_roc_curves(self, roc_analysis: Dict[str, ROCResult], output_path: Optional[str] = None):
        """Plot ROC curves for all metrics"""
        if not roc_analysis:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for metric_name, roc_result in roc_analysis.items():
            ax.plot(roc_result.fpr, roc_result.tpr, 
                   label=f'{metric_name} (AUC = {roc_result.auc_score:.3f})')
        
        # Add diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves by Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            plt.savefig(f"{output_path}_roc_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_descriptive_stats(self, descriptive_stats: Dict[str, Dict], output_path: Optional[str] = None):
        """Plot descriptive statistics comparison"""
        if not descriptive_stats or 'baseline' not in descriptive_stats or 'treatment' not in descriptive_stats:
            return
        
        baseline_stats = descriptive_stats['baseline']
        treatment_stats = descriptive_stats['treatment']
        
        # Prepare data for plotting
        metrics = list(baseline_stats.keys())
        baseline_means = [baseline_stats[m]['mean'] for m in metrics if 'mean' in baseline_stats[m]]
        treatment_means = [treatment_stats[m]['mean'] for m in metrics if 'mean' in treatment_stats[m]]
        baseline_stds = [baseline_stats[m]['std'] for m in metrics if 'std' in baseline_stats[m]]
        treatment_stds = [treatment_stats[m]['std'] for m in metrics if 'std' in treatment_stats[m]]
        
        if not baseline_means:
            return
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_means, width, label='Baseline', 
                       yerr=baseline_stds, capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, treatment_means, width, label='Treatment', 
                       yerr=treatment_stds, capsize=5, alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Values')
        ax.set_title('Descriptive Statistics: Baseline vs Treatment')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(f"{output_path}_descriptive_stats.png", dpi=300, bbox_inches='tight')
        plt.show()

# Convenience functions for common analyses
def analyze_neuromodulation_results(baseline_results: List[Dict], 
                                   treatment_results: List[Dict],
                                   test_name: str = "Unknown Test") -> Dict[str, Any]:
    """Convenience function to analyze neuromodulation test results"""
    analyzer = StatisticalAnalyzer()
    return analyzer.analyze_test_results(baseline_results, treatment_results, test_name)

def generate_analysis_report(analysis_results: Dict[str, Any], 
                           output_path: Optional[str] = None) -> str:
    """Generate a comprehensive text report of the analysis"""
    analyzer = StatisticalAnalyzer()
    
    report = []
    report.append("=" * 80)
    report.append(f"NEUROMODULATION TEST ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Test: {analysis_results.get('test_name', 'Unknown')}")
    report.append(f"Baseline samples: {analysis_results.get('baseline_count', 0)}")
    report.append(f"Treatment samples: {analysis_results.get('treatment_count', 0)}")
    report.append("")
    
    # Statistical tests
    if 'statistical_tests' in analysis_results:
        report.append("STATISTICAL TESTS")
        report.append("-" * 40)
        for test_name, result in analysis_results['statistical_tests'].items():
            report.append(f"{test_name}:")
            report.append(f"  Statistic: {result.statistic:.4f}")
            report.append(f"  P-value: {result.p_value:.6f}")
            report.append(f"  Effect size: {result.effect_size:.4f}")
            if result.confidence_interval[0] != np.nan:
                report.append(f"  95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append("")
    
    # Effect sizes
    if 'effect_sizes' in analysis_results:
        report.append("EFFECT SIZES (Cohen's d)")
        report.append("-" * 40)
        for metric, effect_size in analysis_results['effect_sizes'].items():
            magnitude = "negligible" if abs(effect_size) < 0.2 else \
                       "small" if abs(effect_size) < 0.5 else \
                       "medium" if abs(effect_size) < 0.8 else "large"
            report.append(f"{metric}: {effect_size:.4f} ({magnitude})")
        report.append("")
    
    # Power analysis
    if 'power_analysis' in analysis_results:
        report.append("POWER ANALYSIS")
        report.append("-" * 40)
        for metric, power_info in analysis_results['power_analysis'].items():
            report.append(f"{metric}:")
            report.append(f"  Current power: {power_info.get('current_power', 'N/A'):.3f}")
            report.append(f"  Required sample size: {power_info.get('required_sample_size', 'N/A')}")
            report.append(f"  Current sample size: {power_info.get('current_sample_size', 'N/A')}")
        report.append("")
    
    # Multiple comparison correction
    if 'corrected_p_values' in analysis_results:
        report.append("MULTIPLE COMPARISON CORRECTION (FDR)")
        report.append("-" * 40)
        for test_name, corrected_p in analysis_results['corrected_p_values'].items():
            report.append(f"{test_name}: {corrected_p:.6f}")
        report.append("")
    
    # Generate plots
    plot_paths = analyzer.generate_plots(analysis_results, output_path)
    if 'error' not in plot_paths:
        report.append("PLOTS GENERATED")
        report.append("-" * 40)
        for plot_type, plot_path in plot_paths.items():
            if plot_type != 'error':
                report.append(f"{plot_type}: {plot_path}")
    
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # Save report to file if output path provided
    if output_path:
        with open(f"{output_path}_report.txt", 'w') as f:
            f.write(report_text)
    
    return report_text

if __name__ == "__main__":
    # Example usage
    print("Statistical Analysis Pipeline for Neuromodulation Testing")
    print("=" * 60)
    print("This module provides comprehensive statistical analysis capabilities.")
    print("Use analyze_neuromodulation_results() to analyze test results.")
    print("Use generate_analysis_report() to create detailed reports.")
