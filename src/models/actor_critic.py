# src/models/actor_critic.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_size):
        """
        Actor-Critic Network for PPO.

        Args:
            obs_shape (tuple): Shape of the observation space.
            action_size (int): Number of discrete actions.
        """
        super().__init__()

        # Feature extractor (a simple MLP)
        self.feature_extractor = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head: outputs action probabilities
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

        # Critic head: outputs a state value
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        """
        Forward pass through the network.
        
        Args:
            obs (torch.Tensor): Batch of observations.

        Returns:
            tuple: (action_logits, state_value)
        """
        features = self.feature_extractor(obs)
        action_logits = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_logits, state_value

    def get_action_and_value(self, obs, action=None):
        """
        Get action, log probability, entropy, and state value from an observation.

        Args:
            obs (torch.Tensor): A single observation or a batch of observations.
            action (torch.Tensor, optional): A specific action to evaluate. Defaults to None.

        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        logits, value = self.forward(obs)
        
        # Create a categorical distribution from the logits
        probs = Categorical(logits=logits)
        
        # If no action is provided, sample a new one
        if action is None:
            action = probs.sample()
            
        # Calculate log probability of the action and entropy of the distribution
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1) # Squeeze value to remove last dim
