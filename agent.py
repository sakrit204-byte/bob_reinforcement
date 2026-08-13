"""
Double Dueling DQN Agent for Bob's World 3D.
Features Polyak target network synchronization, experience replay, smooth epsilon decay schedule,
and strict action-masking for grounded jumping.
"""

import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from models import DuelingDQN

class ReplayBuffer:
    """Experience Replay Buffer for storing transitions."""
    def __init__(self, capacity=config.BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Double Dueling DQN Agent with Target Network & Polyak Soft Updates."""
    def __init__(self, state_dim=config.STATE_DIM, action_dim=config.ACTION_DIM):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy & Target Networks
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer & Replay Memory
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stable gradients
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        
        # Epsilon Exploration Parameters
        self.epsilon = config.EPSILON_START
        self.epsilon_min = config.EPSILON_END
        self.epsilon_decay = (config.EPSILON_START - config.EPSILON_END) / config.EPSILON_DECAY_EPISODES
        
        self.train_steps = 0
        self.last_actions = deque(maxlen=10)
        self.tau = 0.01  # Polyak soft update parameter
        self.latest_activations = {}
    
    def reset_agent_weights(self):
        """Resets policy and target neural network weights to pure random initialization (Tabula Rasa)."""
        self.policy_net.reset_parameters()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LR)
        self.memory = ReplayBuffer(config.BUFFER_SIZE)
        self.epsilon = config.EPSILON_START
        self.train_steps = 0
        print("\n  [Model Reset]: Neural network weights wiped to PURE UNTRAINED RANDOM initialization!")
    
    def update_epsilon(self):
        """Decay epsilon at the end of each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
    
    def act(self, state, stuck=False, on_ground=True, can_jump=True, eval_mode=False):
        """
        Select action using epsilon-greedy policy across 5 3D Spatial Actions.
        Captures live layer activations (h1, h2, Q-values) for HUD visualization.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.policy_net.eval()
        with torch.no_grad():
            q_values_tensor, activations = self.policy_net(state_tensor, return_activations=True)
            q_values = q_values_tensor.squeeze(0).clone()
            self.latest_activations = {
                'input': state,
                'h1': activations['hidden1'].cpu().numpy()[0],
                'h2': activations['hidden2'].cpu().numpy()[0],
                'q_values': q_values.cpu().numpy()
            }
        self.policy_net.train()

        # Determine allowed actions
        allowed_actions = [0, 1, 2, 3]
        if on_ground and can_jump:
            allowed_actions.append(4)

        if stuck:
            # If stuck, choose random movement from allowed actions, avoiding jumping to get unstuck faster
            unstuck_actions = [a for a in allowed_actions if a != 4]
            if not unstuck_actions:
                unstuck_actions = [0, 1, 2, 3]
            act_idx = random.choice(unstuck_actions)
            self.last_actions.append(act_idx)
            if hasattr(self, 'latest_activations'):
                self.latest_activations['best_action'] = act_idx
            return act_idx
        
        current_eps = 0.0 if eval_mode else self.epsilon
        
        if random.random() < current_eps:
            act_idx = random.choice(allowed_actions)
        else:
            # Exploitation: mask out forbidden actions
            for a in range(5):
                if a not in allowed_actions:
                    q_values[a] = -float('inf')
            act_idx = torch.argmax(q_values).item()
        
        self.last_actions.append(act_idx)
        if hasattr(self, 'latest_activations'):
            self.latest_activations['best_action'] = act_idx
        return act_idx
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self, batch_size=config.BATCH_SIZE):
        """Perform a Double DQN optimization step over a mini-batch with Polyak target update."""
        if len(self.memory) < batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Current Q-values from policy network
        current_q = self.policy_net(states_t).gather(1, actions_t)
        
        # Double DQN target calculation:
        # 1. Action selection using Policy Net: argmax_a Q_policy(s', a)
        # 2. Action evaluation using Target Net: Q_target(s', best_action)
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * config.GAMMA * next_q
        
        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        
        # Polyak Soft Target Network Update for smooth fast adaptation
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)
            
        return loss.item()
