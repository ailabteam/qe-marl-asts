# src/models/actor_critic.py

import torch
import torch.nn as nn

class Q_ActorCritic(nn.Module):
    def __init__(self, obs_shape, state_shape, n_agents, n_tasks):
        """
        Actor-Critic Network for Q-MAPPO.
        The Actor now outputs a utility matrix.

        Args:
            obs_shape (tuple): Shape of a single agent's local observation.
            state_shape (tuple): Shape of the global state.
            n_agents (int): Number of agents.
            n_tasks (int): Number of tasks. The action space size is n_tasks + 1 (for 'idle').
        """
        super().__init__()
        self.n_agents = n_agents
        self.action_size = n_tasks + 1

        # Actor: Takes stacked local observations and outputs a utility matrix
        # Input shape: (batch_size, n_agents * obs_dim)
        # Output shape: (batch_size, n_agents, action_size)
        self.actor = nn.Sequential(
            nn.Linear(n_agents * obs_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_agents * self.action_size) # Output flat utility matrix
        )

        # Critic: Processes global state (same as MAPPO)
        self.critic = nn.Sequential(
            nn.Linear(state_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def get_utilities(self, stacked_obs):
        """
        Get utility matrix from the actor network.
        
        Args:
            stacked_obs (torch.Tensor): Batch of stacked observations from all agents.
                                       Shape: (batch_size, n_agents * obs_dim)

        Returns:
            torch.Tensor: Utility matrix of shape (batch_size, n_agents, action_size)
        """
        flat_utilities = self.actor(stacked_obs)
        # Reshape to (batch_size, n_agents, action_size)
        utilities = flat_utilities.view(-1, self.n_agents, self.action_size)
        return utilities

    def get_value(self, state):
        """
        Get state value from the critic network based on global state.
        """
        return self.critic(state)
