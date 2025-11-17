# src/models/actor_critic.py

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, state_shape, action_size):
        """
        Actor-Critic Network for MAPPO.
        The Critic now takes the global state as input.

        Args:
            obs_shape (tuple): Shape of the agent's local observation.
            state_shape (tuple): Shape of the global state.
            action_size (int): Number of discrete actions.
        """
        super().__init__()

        # Actor: Processes local observations
        self.actor = nn.Sequential(
            nn.Linear(obs_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

        # Critic: Processes global state
        self.critic = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_action(self, obs):
        """
        Get action from the actor network based on local observation.
        
        Args:
            obs (torch.Tensor): Batch of local observations.

        Returns:
            tuple: (action, log_prob, entropy)
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
    
    def get_logprob_and_entropy(self, obs, action):
        """
        Get log probability and entropy for a given action.
        """
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy()

    def get_value(self, state):
        """
        Get state value from the critic network based on global state.

        Args:
            state (torch.Tensor): Batch of global states.

        Returns:
            torch.Tensor: State values.
        """
        return self.critic(state)
