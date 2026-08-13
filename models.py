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
        """Initializes linear layers using Kaiming Uniform initialization for stable Deep Q-learning."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, state, return_activations=False):
        # Layer 1
        x1 = self.feature[0](state)
        x1_norm = self.feature[1](x1)
        a1 = self.feature[2](x1_norm)
        
        # Layer 2
        x2 = self.feature[3](a1)
        x2_norm = self.feature[4](x2)
        a2 = self.feature[5](x2_norm)
        
        # Value Stream
        v1 = self.value_stream[0](a2)
        v1_act = self.value_stream[1](v1)
        val = self.value_stream[2](v1_act)
        
        # Advantage Stream
        adv1 = self.advantage_stream[0](a2)
        adv1_act = self.advantage_stream[1](adv1)
        adv = self.advantage_stream[2](adv1_act)
        
        # Combine V(s) and A(s, a) with mean subtraction
        q_values = val + (adv - adv.mean(dim=-1, keepdim=True))
        
        if return_activations:
            activations = {
                'input': state,
                'hidden1': a1,
                'hidden2': a2,
                'value': val,
                'advantage': adv,
                'q_values': q_values
            }
            return q_values, activations
            
        return q_values
