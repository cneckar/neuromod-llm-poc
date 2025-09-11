"""
Reinforcement Learning Optimization for Pack Parameters

Implements policy gradient methods for optimizing pack parameters
using RL with the emotion system as reward signal.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class RLOptimizationConfig:
    """Configuration for RL optimization"""
    n_episodes: int = 100
    episode_length: int = 10
    learning_rate: float = 0.01
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Entropy regularization
    value_coef: float = 0.5  # Value function loss coefficient
    hidden_size: int = 64
    batch_size: int = 32
    clip_ratio: float = 0.2  # PPO clipping ratio
    random_seed: Optional[int] = None

class PolicyNetwork(nn.Module):
    """Neural network policy for pack parameter optimization"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class ValueNetwork(nn.Module):
    """Value network for estimating state values"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)

class RLOptimizer:
    """
    Reinforcement Learning optimizer using policy gradient methods.
    
    This implementation uses a simplified PPO (Proximal Policy Optimization)
    approach to optimize pack parameters based on emotion system rewards.
    """
    
    def __init__(self, config: RLOptimizationConfig = None):
        self.config = config or RLOptimizationConfig()
        self.policy_net = None
        self.value_net = None
        self.policy_optimizer = None
        self.value_optimizer = None
        
        # Experience buffer
        self.experiences = deque(maxlen=1000)
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            torch.manual_seed(self.config.random_seed)
    
    def optimize(self, 
                objective_function: Callable[[np.ndarray], float],
                bounds: List[Tuple[float, float]],
                n_episodes: int = None) -> Tuple[np.ndarray, float, List[float]]:
        """
        Run RL optimization.
        
        Args:
            objective_function: Function to minimize (will be negated for reward)
            bounds: List of (min, max) bounds for each parameter
            n_episodes: Number of training episodes
            
        Returns:
            Tuple of (best_parameters, best_value, history)
        """
        n_episodes = n_episodes or self.config.n_episodes
        n_params = len(bounds)
        
        logger.info(f"Starting RL optimization with {n_params} parameters, {n_episodes} episodes")
        
        # Initialize networks
        self._initialize_networks(n_params)
        
        # Convert bounds to normalized [0, 1] space
        bounds_min = np.array([b[0] for b in bounds])
        bounds_range = np.array([b[1] - b[0] for b in bounds])
        
        best_params = None
        best_value = float('inf')
        history = []
        
        # Training loop
        for episode in range(n_episodes):
            episode_rewards = []
            episode_params = []
            episode_values = []
            
            # Generate episode
            for step in range(self.config.episode_length):
                # Get current state (simplified: use episode/step as state)
                state = self._get_state(episode, step, n_episodes)
                
                # Get action from policy
                action = self._get_action(state, bounds_min, bounds_range)
                
                # Evaluate objective (negate for reward)
                reward = -objective_function(action)
                
                # Store experience
                value = self._get_value(state)
                self.experiences.append({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'value': value
                })
                
                episode_rewards.append(reward)
                episode_params.append(action)
                episode_values.append(value)
            
            # Update networks
            if len(self.experiences) >= self.config.batch_size:
                self._update_networks()
            
            # Track best
            episode_avg_reward = np.mean(episode_rewards)
            episode_best_idx = np.argmax(episode_rewards)
            episode_best_params = episode_params[episode_best_idx]
            episode_best_value = -episode_rewards[episode_best_idx]
            
            if episode_best_value < best_value:
                best_value = episode_best_value
                best_params = episode_best_params.copy()
            
            history.append(episode_avg_reward)
            
            if episode % 10 == 0:
                logger.info(f"Episode {episode}: Avg reward = {episode_avg_reward:.4f}, Best value = {best_value:.4f}")
        
        logger.info(f"RL optimization complete. Best value: {best_value:.4f}")
        
        return best_params, best_value, history
    
    def _initialize_networks(self, n_params: int):
        """Initialize policy and value networks"""
        # State size: episode progress + step progress + parameter count
        state_size = 3  # Simplified state representation
        
        self.policy_net = PolicyNetwork(state_size, self.config.hidden_size, n_params)
        self.value_net = ValueNetwork(state_size, self.config.hidden_size)
        
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.config.learning_rate)
    
    def _get_state(self, episode: int, step: int, total_episodes: int) -> np.ndarray:
        """Get current state representation"""
        episode_progress = episode / total_episodes
        step_progress = step / self.config.episode_length
        param_count = 1.0  # Normalized parameter count
        
        return np.array([episode_progress, step_progress, param_count], dtype=np.float32)
    
    def _get_action(self, state: np.ndarray, bounds_min: np.ndarray, bounds_range: np.ndarray) -> np.ndarray:
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_logits = self.policy_net(state_tensor)
            # Convert to probabilities using softmax
            action_probs = torch.softmax(action_logits, dim=-1)
            # Sample action
            action = torch.multinomial(action_probs, 1).squeeze().numpy()
        
        # Convert to actual parameter values
        # Map [0, 1] to parameter bounds
        normalized_action = action / (action_logits.shape[-1] - 1)  # Normalize to [0, 1]
        actual_params = bounds_min + normalized_action * bounds_range
        
        return actual_params
    
    def _get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            value = self.value_net(state_tensor).squeeze().item()
        
        return value
    
    def _update_networks(self):
        """Update policy and value networks using PPO"""
        if len(self.experiences) < self.config.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.experiences, min(self.config.batch_size, len(self.experiences)))
        
        states = torch.FloatTensor([exp['state'] for exp in batch])
        actions = torch.FloatTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])
        old_values = torch.FloatTensor([exp['value'] for exp in batch])
        
        # Compute returns
        returns = self._compute_returns(rewards)
        
        # Update value network
        self._update_value_network(states, returns)
        
        # Update policy network
        self._update_policy_network(states, actions, returns, old_values)
    
    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute discounted returns"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def _update_value_network(self, states: torch.Tensor, returns: torch.Tensor):
        """Update value network"""
        self.value_optimizer.zero_grad()
        
        values = self.value_net(states).squeeze()
        value_loss = nn.MSELoss()(values, returns)
        
        value_loss.backward()
        self.value_optimizer.step()
    
    def _update_policy_network(self, 
                              states: torch.Tensor, 
                              actions: torch.Tensor, 
                              returns: torch.Tensor, 
                              old_values: torch.Tensor):
        """Update policy network using PPO"""
        self.policy_optimizer.zero_grad()
        
        # Get current policy logits
        current_logits = self.policy_net(states)
        current_probs = torch.softmax(current_logits, dim=-1)
        
        # Compute advantages
        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss (simplified PPO)
        # For simplicity, we'll use a basic policy gradient approach
        log_probs = torch.log(current_probs + 1e-8)
        policy_loss = -(log_probs * advantages.unsqueeze(-1)).mean()
        
        # Add entropy regularization
        entropy = -(current_probs * log_probs).sum(dim=-1).mean()
        total_loss = policy_loss - self.config.entropy_coef * entropy
        
        total_loss.backward()
        self.policy_optimizer.step()

# Convenience function
def rl_optimize(objective_function: Callable[[np.ndarray], float],
               bounds: List[Tuple[float, float]],
               n_episodes: int = 100,
               config: RLOptimizationConfig = None) -> Tuple[np.ndarray, float, List[float]]:
    """Convenience function for RL optimization"""
    optimizer = RLOptimizer(config)
    return optimizer.optimize(objective_function, bounds, n_episodes)
