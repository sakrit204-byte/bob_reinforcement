"""
PyTorch Neural Network Models for Bob's World 3D RL Agent.
Implements Dueling Deep Q-Network (Dueling DQN) with DUMB/BLIND initial weights (Normal 0, 0.01).
Bob starts completely un-biased with zero prior knowledge of movement or room layout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network Architecture with DUMB/BLIND Initial Weights.
    """
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Shared Feature Extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Action Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes linear layers with small Gaussian noise N(0, 0.01) so Bob starts completely dumb and un-biased."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, state):
        features = self.feature(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine V(s) and A(s, a) with mean subtraction
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values
