# src/algos/ppo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.actor_critic import ActorCritic

class PPOAgent:
    def __init__(self, obs_shape, action_size, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5):
        """
        PPO Agent.

        Args:
            obs_shape (tuple): Shape of the observation space.
            action_size (int): Number of discrete actions.
            device (torch.device): Device to run the computations on (cpu or cuda).
            lr (float): Learning rate.
            gamma (float): Discount factor.
            gae_lambda (float): Lambda for Generalized Advantage Estimation (GAE).
            clip_coef (float): PPO clipping coefficient.
            ent_coef (float): Entropy coefficient for encouraging exploration.
            vf_coef (float): Value function loss coefficient.
        """
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.network = ActorCritic(obs_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_action_and_value(self, obs, action=None):
        """Wrapper for the network's method."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return self.network.get_action_and_value(obs, action)

    def compute_advantages(self, rewards, dones, values, next_value):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        """
        num_steps = len(rewards)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_val = values[t+1]

            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
        
        returns = advantages + values
        return advantages, returns

    def learn(self, batch_obs, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        """
        Update the policy and value function.
        """
        # Convert numpy arrays to torch tensors
        b_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
        b_actions = torch.tensor(batch_actions, dtype=torch.int64).to(self.device)
        b_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(self.device)
        b_advantages = torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)
        b_returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)

        # Get new logprobs, entropy, and values from the network
        _, new_logprob, entropy, new_value = self.network.get_action_and_value(b_obs, b_actions)
        
        logratio = new_logprob - b_logprobs
        ratio = logratio.exp()

        # Policy loss (PPO-Clip objective)
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Total loss
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Clip gradients
        self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()
