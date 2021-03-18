import copy
import torch
import config
import random
import numpy as np


def concat(x):
    """Concat action or state
    """
    return np.array(x).reshape(1,-1).squeeze()

def split(size, id_agent, x):
    """Split action or state
    """
    list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(config.DEVICE)    
    return x.index_select(1, list_indices)

def soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process.
        Params
        ======
            size (int) : size of action space
            target_model: PyTorch model (weights will be copied to)
            mu (float) :  Ornstein-Uhlenbeck noise parameter
            theta (float) :  Ornstein-Uhlenbeck noise parameter
            sigma (flmoat) : Ornstein-Uhlenbeck noise parameter 
        """
    def __init__(self, size, seed, mu=config.MU, theta=config.THETA, sigma=config.SIGMA):
        """Initialize parameters and noise process."""
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state
