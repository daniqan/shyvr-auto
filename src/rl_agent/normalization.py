"""
Observation Normalization Utilities

Implements RunningMeanStd for dynamic feature normalization in RL agents.
Essential for handling non-stationary market data distributions.
"""

import numpy as np
import torch
from typing import Union, Optional

class RunningMeanStd:
    """
    Tracks the running mean and standard deviation of a data stream.
    Used to normalize observations to N(0, 1) dynamically.
    """
    def __init__(self, shape: tuple, epsilon: float = 1e-4, clip: float = 10.0):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x: np.ndarray):
        """
        Update running statistics with a new batch of data.
        Args:
            x: Batch of data with shape matching self.shape
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        """Update using moments from a batch"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize the input x using current mean and std.
        Args:
            x: Input data (numpy array or torch tensor)
        Returns:
            Normalized data
        """
        is_tensor = isinstance(x, torch.Tensor)
        if is_tensor:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        # Normalize
        normalized = (x_np - self.mean) / np.sqrt(self.var + self.epsilon)
        normalized = np.clip(normalized, -self.clip, self.clip)
        
        if is_tensor:
            return torch.as_tensor(normalized, device=x.device, dtype=x.dtype)
        return normalized
