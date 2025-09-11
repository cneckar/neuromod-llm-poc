#!/usr/bin/env python3
"""
Advanced Statistical Analysis System

This module implements advanced statistical methods for neuromodulation research,
including mixed-effects models, Bayesian hierarchical models, and canonical
correlation analysis as required by the paper.

Key Features:
- Mixed-effects models with random intercepts
- Bayesian hierarchical models with credible intervals
- Canonical correlation analysis for human-model signature matching
- Model comparison and selection (BIC/AIC)
- Statistical significance testing
- Comprehensive result reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import warnings

# Statistical analysis imports
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import svd

# Optional imports for advanced statistics
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import mixedlm
    from statsmodels.stats.multitest import multipletests
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Mixed-effects models will be limited.")

# Bayesian analysis (if available)
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC and ArviZ not available. Bayesian analysis will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MixedEffectsResult:
    """Results from mixed-effects model analysis"""
    model_name: str
    formula: str
    fixed_effects: Dict[str, float]
    fixed_effects_se: Dict[str, float]
    fixed_effects_pvalues: Dict[str, float]
    fixed_effects_ci: Dict[str, Tuple[float, float]]
    random_effects: Dict[str, float]
    random_effects_se: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    n_observations: int
    n_groups: int
    convergence_warning: bool
    model_summary: str

@dataclass
class BayesianResult:
    """Results from Bayesian hierarchical model analysis"""
    model_name: str
    posterior_samples: Dict[str, np.ndarray]
    credible_intervals: Dict[str, Tuple[float, float]]
    posterior_means: Dict[str, float]
    posterior_stds: Dict[str, float]
    effective_sample_size: Dict[str, int]
    rhat: Dict[str, float]
    waic: float
    loo: float
    model_comparison: Dict[str, float]
    trace_summary: str

@dataclass
class CanonicalCorrelationResult:
    """Results from canonical correlation analysis"""
    n_canonical_variates: int
    canonical_correlations: np.ndarray
    canonical_correlations_pvalues: np.ndarray
    canonical_correlations_ci: List[Tuple[float, float]]
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray
    redundancy_x: float
    redundancy_y: float
    significance_test: Dict[str, Any]
    interpretation: str

class AdvancedStatisticalAnalyzer:
    """Main class for advanced statistical analysis"""
    
    def __init__(self, output_dir: str = "outputs/analysis/statistical"):
        """
        Initialize the advanced statistical analyzer
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up statistical parameters
        self.alpha = 0.05
        self.ci_level = 0.95
        self.random_seed = 42
        
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        if BAYESIAN_AVAILABLE:
            pm.set_tt_rng(self.random_seed)
    
    def fit_mixed_effects_model(self, 
                              data: pd.DataFrame,
                              formula: str,
                              group_var: str,
                              model_name: str = "mixed_effects") -> MixedEffectsResult:
        """
        Fit a mixed-effects model with random intercepts
        
        Args:
            data: DataFrame with the data
            formula: Model formula (e.g., "score ~ condition + (1|prompt_set)")
            group_var: Grouping variable for random effects
            model_name: Name for the model
            
        Returns:
            MixedEffectsResult object with model results
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for mixed-effects models. Please install it with: pip install statsmodels")
        
        try:
            # Fit the mixed-effects model
            model = mixedlm(formula, data, groups=data[group_var])
            result = model.fit()
            
            # Extract fixed effects
            fixed_effects = {}
            fixed_effects_se = {}
            fixed_effects_pvalues = {}
            fixed_effects_ci = {}
            
            for param in result.params.index:
                if not param.startswith('Group Var'):
                    fixed_effects[param] = result.params[param]
                    fixed_effects_se[param] = result.bse[param]
                    fixed_effects_pvalues[param] = result.pvalues[param]
                    
                    # Calculate confidence intervals
                    ci_lower = result.params[param] - 1.96 * result.bse[param]
                    ci_upper = result.params[param] + 1.96 * result.bse[param]
                    fixed_effects_ci[param] = (ci_lower, ci_upper)
            
            # Extract random effects
            random_effects = {}
            random_effects_se = {}
            
            for param in result.params.index:
                if param.startswith('Group Var'):
                    random_effects[param] = result.params[param]
                    random_effects_se[param] = result.bse[param]
            
            # Model fit statistics
            aic = result.aic
            bic = result.bic
            log_likelihood = result.llf
            n_observations = result.nobs
            n_groups = len(data[group_var].unique())
            
            # Check for convergence warnings
            convergence_warning = result.mle_retvals.get('converged', True) == False
            
            # Create model summary
            model_summary = str(result.summary())
            
            return MixedEffectsResult(
                model_name=model_name,
                formula=formula,
                fixed_effects=fixed_effects,
                fixed_effects_se=fixed_effects_se,
                fixed_effects_pvalues=fixed_effects_pvalues,
                fixed_effects_ci=fixed_effects_ci,
                random_effects=random_effects,
                random_effects_se=random_effects_se,
                aic=aic,
                bic=bic,
                log_likelihood=log_likelihood,
                n_observations=n_observations,
                n_groups=n_groups,
                convergence_warning=convergence_warning,
                model_summary=model_summary
            )
            
        except Exception as e:
            logger.error(f"Error fitting mixed-effects model: {e}")
            raise
    
    def fit_bayesian_hierarchical_model(self,
                                      data: pd.DataFrame,
                                      y_var: str,
                                      x_vars: List[str],
                                      group_var: str,
                                      model_name: str = "bayesian_hierarchical") -> Optional[BayesianResult]:
        """
        Fit a Bayesian hierarchical model
        
        Args:
            data: DataFrame with the data
            y_var: Dependent variable name
            x_vars: List of independent variable names
            group_var: Grouping variable for hierarchical structure
            model_name: Name for the model
            
        Returns:
            BayesianResult object with model results, or None if Bayesian analysis unavailable
        """
        if not BAYESIAN_AVAILABLE:
            logger.warning("Bayesian analysis not available. Install PyMC and ArviZ for full functionality.")
            return None
        
        try:
            # Prepare data
            y = data[y_var].values
            X = data[x_vars].values
            groups = data[group_var].values
            group_ids = pd.Categorical(groups).codes
            
            n_obs = len(y)
            n_groups = len(np.unique(group_ids))
            n_vars = len(x_vars)
            
            with pm.Model() as model:
                # Priors for fixed effects
                beta = pm.Normal('beta', mu=0, sigma=1, shape=n_vars)
                
                # Priors for random effects (group-level)
                sigma_group = pm.HalfNormal('sigma_group', sigma=1)
                alpha_group = pm.Normal('alpha_group', mu=0, sigma=sigma_group, shape=n_groups)
                
                # Priors for error term
                sigma_error = pm.HalfNormal('sigma_error', sigma=1)
                
                # Linear predictor
                mu = alpha_group[group_ids] + pm.math.dot(X, beta)
                
                # Likelihood
                y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_error, observed=y)
                
                # Sample from posterior
                trace = pm.sample(2000, tune=1000, random_seed=self.random_seed, 
                                progressbar=False, return_inferencedata=True)
            
            # Extract posterior samples
            posterior_samples = {}
            for var in ['beta', 'alpha_group', 'sigma_group', 'sigma_error']:
                posterior_samples[var] = trace.posterior[var].values.flatten()
            
            # Calculate credible intervals
            credible_intervals = {}
            for var, samples in posterior_samples.items():
                ci_lower = np.percentile(samples, (1 - self.ci_level) / 2 * 100)
                ci_upper = np.percentile(samples, (1 + self.ci_level) / 2 * 100)
                credible_intervals[var] = (ci_lower, ci_upper)
            
            # Calculate posterior means and standard deviations
            posterior_means = {var: np.mean(samples) for var, samples in posterior_samples.items()}
            posterior_stds = {var: np.std(samples) for var, samples in posterior_samples.items()}
            
            # Calculate effective sample size and R-hat
            effective_sample_size = {}
            rhat = {}
            for var in trace.posterior.data_vars:
                ess = az.ess(trace, var=var).values
                rhat_val = az.rhat(trace, var=var).values
                effective_sample_size[var] = int(np.mean(ess))
                rhat[var] = float(np.mean(rhat_val))
            
            # Model comparison metrics
            waic = az.waic(trace).waic
            loo = az.loo(trace).loo
            
            # Model comparison
            model_comparison = {
                'waic': waic,
                'loo': loo,
                'effective_sample_size': effective_sample_size,
                'rhat': rhat
            }
            
            # Create trace summary
            trace_summary = str(az.summary(trace))
            
            return BayesianResult(
                model_name=model_name,
                posterior_samples=posterior_samples,
                credible_intervals=credible_intervals,
                posterior_means=posterior_means,
                posterior_stds=posterior_stds,
                effective_sample_size=effective_sample_size,
                rhat=rhat,
                waic=waic,
                loo=loo,
                model_comparison=model_comparison,
                trace_summary=trace_summary
            )
            
        except Exception as e:
            logger.error(f"Error fitting Bayesian hierarchical model: {e}")
            return None
    
    def canonical_correlation_analysis(self,
                                     x_data: np.ndarray,
                                     y_data: np.ndarray,
                                     x_names: List[str] = None,
                                     y_names: List[str] = None) -> CanonicalCorrelationResult:
        """
        Perform canonical correlation analysis
        
        Args:
            x_data: First set of variables (e.g., model signatures)
            y_data: Second set of variables (e.g., human signatures)
            x_names: Names for x variables
            y_names: Names for y variables
            
        Returns:
            CanonicalCorrelationResult object with analysis results
        """
        try:
            # Center the data
            x_centered = x_data - np.mean(x_data, axis=0)
            y_centered = y_data - np.mean(y_data, axis=0)
            
            # Calculate covariance matrices
            n = x_data.shape[0]
            Cxx = np.cov(x_centered.T)
            Cyy = np.cov(y_centered.T)
            Cxy = np.cov(x_centered.T, y_centered.T)[:x_data.shape[1], x_data.shape[1]:]
            
            # Solve the generalized eigenvalue problem
            # Cxx^(-1) * Cxy * Cyy^(-1) * Cyx * a = lambda^2 * a
            Cxx_inv = np.linalg.pinv(Cxx)
            Cyy_inv = np.linalg.pinv(Cyy)
            
            # Calculate the matrix for eigenvalue decomposition
            A = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(A)
            
            # Sort by eigenvalues (canonical correlations)
            idx = np.argsort(eigenvalues)[::-1]
            canonical_correlations = np.sqrt(eigenvalues[idx])
            x_weights = eigenvectors[:, idx]
            
            # Calculate y weights
            y_weights = np.zeros((y_data.shape[1], len(canonical_correlations)))
            for i in range(len(canonical_correlations)):
                y_weights[:, i] = Cyy_inv @ Cxy.T @ x_weights[:, i] / canonical_correlations[i]
            
            # Calculate canonical scores
            x_scores = x_centered @ x_weights
            y_scores = y_centered @ y_weights
            
            # Calculate p-values for canonical correlations
            n_canonical_variates = len(canonical_correlations)
            canonical_correlations_pvalues = np.zeros(n_canonical_variates)
            
            for i in range(n_canonical_variates):
                # Wilks' lambda test
                lambda_val = np.prod(1 - canonical_correlations[i:]**2)
                p = x_data.shape[1]
                q = y_data.shape[1]
                n = x_data.shape[0]
                
                # Approximate chi-square test
                chi2 = -(n - 1 - (p + q + 1) / 2) * np.log(lambda_val)
                df = (p - i) * (q - i)
                canonical_correlations_pvalues[i] = 1 - stats.chi2.cdf(chi2, df)
            
            # Calculate confidence intervals for canonical correlations
            canonical_correlations_ci = []
            for i, r in enumerate(canonical_correlations):
                # Fisher's z transformation
                z = 0.5 * np.log((1 + r) / (1 - r))
                se = 1 / np.sqrt(n - 3)
                z_ci = z + np.array([-1, 1]) * 1.96 * se
                r_ci = (np.exp(2 * z_ci) - 1) / (np.exp(2 * z_ci) + 1)
                canonical_correlations_ci.append((r_ci[0], r_ci[1]))
            
            # Calculate redundancy measures
            # Redundancy of x given y
            redundancy_x = np.sum(canonical_correlations**2) / x_data.shape[1]
            # Redundancy of y given x
            redundancy_y = np.sum(canonical_correlations**2) / y_data.shape[1]
            
            # Significance test
            significance_test = {
                'wilks_lambda': np.prod(1 - canonical_correlations**2),
                'chi_square': -(n - 1 - (x_data.shape[1] + y_data.shape[1] + 1) / 2) * 
                             np.log(np.prod(1 - canonical_correlations**2)),
                'df': x_data.shape[1] * y_data.shape[1],
                'p_value': 1 - stats.chi2.cdf(
                    -(n - 1 - (x_data.shape[1] + y_data.shape[1] + 1) / 2) * 
                    np.log(np.prod(1 - canonical_correlations**2)),
                    x_data.shape[1] * y_data.shape[1]
                )
            }
            
            # Interpretation
            interpretation = self._interpret_canonical_correlation(
                canonical_correlations, canonical_correlations_pvalues, 
                redundancy_x, redundancy_y, significance_test
            )
            
            return CanonicalCorrelationResult(
                n_canonical_variates=n_canonical_variates,
                canonical_correlations=canonical_correlations,
                canonical_correlations_pvalues=canonical_correlations_pvalues,
                canonical_correlations_ci=canonical_correlations_ci,
                x_weights=x_weights,
                y_weights=y_weights,
                x_scores=x_scores,
                y_scores=y_scores,
                redundancy_x=redundancy_x,
                redundancy_y=redundancy_y,
                significance_test=significance_test,
                interpretation=interpretation
            )
            
        except Exception as e:
            logger.error(f"Error in canonical correlation analysis: {e}")
            raise
    
    def _interpret_canonical_correlation(self, 
                                       canonical_correlations: np.ndarray,
                                       p_values: np.ndarray,
                                       redundancy_x: float,
                                       redundancy_y: float,
                                       significance_test: Dict[str, Any]) -> str:
        """Interpret canonical correlation results"""
        n_significant = np.sum(p_values < self.alpha)
        
        interpretation = f"""
        Canonical Correlation Analysis Results:
        
        - Number of significant canonical variates: {n_significant}
        - First canonical correlation: {canonical_correlations[0]:.3f} (p = {p_values[0]:.3f})
        - Redundancy of X given Y: {redundancy_x:.3f}
        - Redundancy of Y given X: {redundancy_y:.3f}
        - Overall significance: {'Significant' if significance_test['p_value'] < self.alpha else 'Not significant'} (p = {significance_test['p_value']:.3f})
        
        Interpretation:
        """
        
        if n_significant > 0:
            interpretation += f"The first {n_significant} canonical variates show significant relationships between the two variable sets. "
            interpretation += f"The first canonical correlation of {canonical_correlations[0]:.3f} indicates a {'strong' if canonical_correlations[0] > 0.7 else 'moderate' if canonical_correlations[0] > 0.5 else 'weak'} relationship. "
        else:
            interpretation += "No significant canonical correlations were found between the variable sets. "
        
        if redundancy_x > 0.1:
            interpretation += f"The redundancy measure ({redundancy_x:.3f}) suggests that the first variable set explains a meaningful amount of variance in the second set. "
        
        return interpretation
    
    def model_comparison(self, 
                        models: List[Union[MixedEffectsResult, BayesianResult]],
                        comparison_type: str = "aic") -> Dict[str, Any]:
        """
        Compare multiple models using information criteria
        
        Args:
            models: List of model results
            comparison_type: Type of comparison ("aic", "bic", "waic", "loo")
            
        Returns:
            Dictionary with comparison results
        """
        comparison_results = {
            'model_names': [],
            'comparison_metric': [],
            'delta_metric': [],
            'weights': [],
            'best_model': None,
            'comparison_summary': ''
        }
        
        if comparison_type == "aic":
            metrics = [model.aic for model in models if hasattr(model, 'aic')]
            model_names = [model.model_name for model in models if hasattr(model, 'aic')]
        elif comparison_type == "bic":
            metrics = [model.bic for model in models if hasattr(model, 'bic')]
            model_names = [model.model_name for model in models if hasattr(model, 'bic')]
        elif comparison_type == "waic" and BAYESIAN_AVAILABLE:
            metrics = [model.waic for model in models if hasattr(model, 'waic')]
            model_names = [model.model_name for model in models if hasattr(model, 'waic')]
        elif comparison_type == "loo" and BAYESIAN_AVAILABLE:
            metrics = [model.loo for model in models if hasattr(model, 'loo')]
            model_names = [model.model_name for model in models if hasattr(model, 'loo')]
        else:
            raise ValueError(f"Invalid comparison type: {comparison_type}")
        
        if not metrics:
            return comparison_results
        
        # Calculate delta values and weights
        min_metric = min(metrics)
        delta_metrics = [metric - min_metric for metric in metrics]
        
        # Calculate Akaike weights
        weights = np.exp(-0.5 * np.array(delta_metrics))
        weights = weights / np.sum(weights)
        
        # Find best model
        best_idx = np.argmin(metrics)
        best_model = model_names[best_idx]
        
        comparison_results.update({
            'model_names': model_names,
            'comparison_metric': metrics,
            'delta_metric': delta_metrics,
            'weights': weights.tolist(),
            'best_model': best_model
        })
        
        # Create summary
        summary = f"Model Comparison Results ({comparison_type.upper()}):\n"
        summary += f"Best model: {best_model}\n\n"
        
        for i, (name, metric, delta, weight) in enumerate(zip(model_names, metrics, delta_metrics, weights)):
            summary += f"{i+1}. {name}: {metric:.3f} (Δ={delta:.3f}, weight={weight:.3f})\n"
        
        comparison_results['comparison_summary'] = summary
        
        return comparison_results
    
    def export_results(self, 
                      results: Union[MixedEffectsResult, BayesianResult, CanonicalCorrelationResult],
                      filename: str = None) -> Path:
        """
        Export analysis results to files
        
        Args:
            results: Analysis results to export
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{results.model_name}_{timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # Convert results to dictionary
        if isinstance(results, MixedEffectsResult):
            results_dict = asdict(results)
        elif isinstance(results, BayesianResult):
            # Convert numpy arrays to lists for JSON serialization
            results_dict = asdict(results)
            for key, value in results_dict.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, np.ndarray):
                            results_dict[key][subkey] = subvalue.tolist()
                elif isinstance(value, np.ndarray):
                    results_dict[key] = value.tolist()
        elif isinstance(results, CanonicalCorrelationResult):
            results_dict = asdict(results)
            for key, value in results_dict.items():
                if isinstance(value, np.ndarray):
                    results_dict[key] = value.tolist()
        
        # Save to JSON
        import json
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        logger.info(f"Results exported to: {output_path}")
        return output_path


def create_sample_data() -> pd.DataFrame:
    """Create sample data for testing the advanced statistical analysis"""
    np.random.seed(42)
    
    n_obs = 200
    n_prompt_sets = 10
    n_conditions = 3
    
    data = []
    for prompt_set in range(n_prompt_sets):
        for condition in range(n_conditions):
            for obs in range(n_obs // (n_prompt_sets * n_conditions)):
                # Generate data with mixed effects
                prompt_effect = np.random.normal(0, 0.5)
                condition_effect = condition * 0.3 + np.random.normal(0, 0.1)
                error = np.random.normal(0, 0.2)
                
                score = 0.5 + prompt_effect + condition_effect + error
                
                data.append({
                    'prompt_set': f'set_{prompt_set}',
                    'condition': f'condition_{condition}',
                    'score': score,
                    'seed': np.random.randint(1, 1000)
                })
    
    return pd.DataFrame(data)


def main():
    """Main function to demonstrate advanced statistical analysis"""
    print("📊 Advanced Statistical Analysis Demo")
    print("=" * 50)
    
    # Create analyzer
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Create sample data
    print("📈 Creating sample data...")
    data = create_sample_data()
    print(f"✅ Created dataset with {len(data)} observations")
    
    # Mixed-effects model
    print("\n🔧 Fitting mixed-effects model...")
    try:
        me_result = analyzer.fit_mixed_effects_model(
            data=data,
            formula="score ~ condition + (1|prompt_set)",
            group_var="prompt_set",
            model_name="sample_mixed_effects"
        )
        print("✅ Mixed-effects model fitted successfully")
        print(f"   AIC: {me_result.aic:.3f}, BIC: {me_result.bic:.3f}")
        print(f"   Fixed effects: {me_result.fixed_effects}")
    except Exception as e:
        print(f"❌ Error fitting mixed-effects model: {e}")
    
    # Bayesian hierarchical model
    print("\n🔮 Fitting Bayesian hierarchical model...")
    try:
        bayes_result = analyzer.fit_bayesian_hierarchical_model(
            data=data,
            y_var="score",
            x_vars=["condition"],
            group_var="prompt_set",
            model_name="sample_bayesian"
        )
        if bayes_result:
            print("✅ Bayesian hierarchical model fitted successfully")
            print(f"   WAIC: {bayes_result.waic:.3f}, LOO: {bayes_result.loo:.3f}")
        else:
            print("⚠️ Bayesian analysis not available (PyMC/ArviZ not installed)")
    except Exception as e:
        print(f"❌ Error fitting Bayesian model: {e}")
    
    # Canonical correlation analysis
    print("\n🔗 Performing canonical correlation analysis...")
    try:
        # Create sample signature data
        n_samples = 50
        model_signatures = np.random.randn(n_samples, 5)
        human_signatures = model_signatures + np.random.randn(n_samples, 5) * 0.3
        
        cca_result = analyzer.canonical_correlation_analysis(
            x_data=model_signatures,
            y_data=human_signatures,
            x_names=[f"model_dim_{i}" for i in range(5)],
            y_names=[f"human_dim_{i}" for i in range(5)]
        )
        print("✅ Canonical correlation analysis completed")
        print(f"   First canonical correlation: {cca_result.canonical_correlations[0]:.3f}")
        print(f"   Significance: {'Yes' if cca_result.canonical_correlations_pvalues[0] < 0.05 else 'No'}")
    except Exception as e:
        print(f"❌ Error in canonical correlation analysis: {e}")
    
    print("\n🎉 Advanced statistical analysis demo complete!")
    print(f"📂 Results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
