"""
Bayesian Optimization for Pack Parameters

Implements Gaussian Process-based Bayesian optimization for finding
optimal pack parameters.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

@dataclass
class BayesianOptimizationConfig:
    """Configuration for Bayesian optimization"""
    n_initial_points: int = 10
    n_iterations: int = 50
    acquisition_function: str = 'ei'  # 'ei', 'pi', 'ucb'
    xi: float = 0.01  # Exploration parameter for EI/PI
    kappa: float = 2.576  # Exploration parameter for UCB
    kernel_length_scale: float = 1.0
    kernel_noise: float = 0.1
    random_seed: Optional[int] = None

class BayesianOptimizer:
    """
    Bayesian optimization using Gaussian Process regression.
    
    This implementation uses scikit-learn's GaussianProcessRegressor
    with acquisition functions for efficient exploration.
    """
    
    def __init__(self, config: BayesianOptimizationConfig = None):
        self.config = config or BayesianOptimizationConfig()
        self.gp = None
        self.X_observed = []
        self.y_observed = []
        self.best_x = None
        self.best_y = float('inf')
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float],
                bounds: List[Tuple[float, float]],
                n_iterations: int = None) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run Bayesian optimization.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) bounds for each parameter
            n_iterations: Number of optimization iterations
            
        Returns:
            Tuple of (best_parameters, best_value, history)
        """
        n_iterations = n_iterations or self.config.n_iterations
        n_params = len(bounds)
        
        logger.info(f"Starting Bayesian optimization with {n_params} parameters, {n_iterations} iterations")
        
        # Initialize with random points
        self._initialize_random_points(objective_function, bounds, self.config.n_initial_points)
        
        # Main optimization loop
        for iteration in range(n_iterations):
            logger.debug(f"Bayesian optimization iteration {iteration + 1}/{n_iterations}")
            
            # Fit Gaussian Process
            self._fit_gp()
            
            # Find next point to evaluate
            next_x = self._acquisition_optimization(bounds)
            
            # Evaluate objective
            next_y = objective_function(next_x)
            
            # Update observations
            self.X_observed.append(next_x)
            self.y_observed.append(next_y)
            
            # Update best
            if next_y < self.best_y:
                self.best_y = next_y
                self.best_x = next_x.copy()
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Best value = {self.best_y:.4f}")
        
        logger.info(f"Bayesian optimization complete. Best value: {self.best_y:.4f}")
        
        return self.best_x, self.best_y, self.y_observed.copy()
    
    def _initialize_random_points(self, 
                                objective_function: Callable,
                                bounds: List[Tuple[float, float]],
                                n_points: int):
        """Initialize with random points"""
        n_params = len(bounds)
        
        for _ in range(n_points):
            # Sample random point within bounds
            x = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_params)])
            y = objective_function(x)
            
            self.X_observed.append(x)
            self.y_observed.append(y)
            
            # Update best
            if y < self.best_y:
                self.best_y = y
                self.best_x = x.copy()
    
    def _fit_gp(self):
        """Fit Gaussian Process to observed data"""
        if len(self.X_observed) < 2:
            return
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        
        # Define kernel
        kernel = (Matern(length_scale=self.config.kernel_length_scale, nu=2.5) + 
                 WhiteKernel(noise_level=self.config.kernel_noise))
        
        # Fit GP
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=self.config.random_seed
        )
        
        try:
            self.gp.fit(X, y)
        except Exception as e:
            logger.warning(f"GP fitting failed: {e}, using simple RBF kernel")
            # Fallback to simple RBF kernel
            self.gp = GaussianProcessRegressor(
                kernel=RBF(length_scale=1.0),
                alpha=1e-6,
                normalize_y=True,
                random_state=self.config.random_seed
            )
            self.gp.fit(X, y)
    
    def _acquisition_optimization(self, bounds: List[Tuple[float, float]]) -> np.ndarray:
        """Find next point to evaluate using acquisition function"""
        if self.gp is None:
            # Fallback to random sampling
            n_params = len(bounds)
            return np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_params)])
        
        def acquisition_function(x):
            x = x.reshape(1, -1)
            mu, sigma = self.gp.predict(x, return_std=True)
            
            if self.config.acquisition_function == 'ei':
                # Expected Improvement
                improvement = self.best_y - mu
                z = improvement / (sigma + 1e-8)
                ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
                return -ei[0]  # Minimize negative EI
                
            elif self.config.acquisition_function == 'pi':
                # Probability of Improvement
                improvement = self.best_y - mu
                z = improvement / (sigma + 1e-8)
                pi = self._normal_cdf(z)
                return -pi[0]  # Minimize negative PI
                
            elif self.config.acquisition_function == 'ucb':
                # Upper Confidence Bound
                ucb = mu + self.config.kappa * sigma
                return ucb[0]  # Minimize UCB
                
            else:
                # Default to EI
                improvement = self.best_y - mu
                z = improvement / (sigma + 1e-8)
                ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
                return -ei[0]
        
        # Optimize acquisition function
        n_params = len(bounds)
        x0 = np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_params)])
        
        try:
            result = minimize(
                acquisition_function,
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 100}
            )
            
            if result.success:
                return result.x
            else:
                # Fallback to random sampling
                return np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_params)])
                
        except Exception as e:
            logger.warning(f"Acquisition optimization failed: {e}, using random sampling")
            return np.array([np.random.uniform(bounds[i][0], bounds[i][1]) for i in range(n_params)])
    
    def _normal_cdf(self, x):
        """Cumulative distribution function of standard normal"""
        return 0.5 * (1 + torch.erf(torch.tensor(x) / np.sqrt(2))).numpy()
    
    def _normal_pdf(self, x):
        """Probability density function of standard normal"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def get_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get GP predictions for given points"""
        if self.gp is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu, sigma

# Convenience function
def bayesian_optimize(objective_function: Callable[[np.ndarray], float],
                     bounds: List[Tuple[float, float]],
                     n_iterations: int = 50,
                     config: BayesianOptimizationConfig = None) -> Tuple[np.ndarray, float, List[float]]:
    """Convenience function for Bayesian optimization"""
    optimizer = BayesianOptimizer(config)
    return optimizer.optimize(objective_function, bounds, n_iterations)
