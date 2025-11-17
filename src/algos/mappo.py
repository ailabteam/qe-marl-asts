# src/algos/mappo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.actor_critic import ActorCritic

class MAPPOAgent:
    def __init__(self, n_agents, obs_shape, state_shape, action_size, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_coef=0.2, ent_coef=0.01, vf_coef=0.5):
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        # All agents share the same network
        self.network = ActorCritic(obs_shape, state_shape, action_size).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            # obs is expected to be shape (n_agents, obs_dim)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        return self.network.get_action(obs)

    def get_value(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.network.get_value(state)

    def compute_advantages(self, rewards, dones, values, next_value):
        # Same as PPO, but expects rewards/dones/values of shape (num_steps, n_agents)
        num_steps = rewards.shape[0]
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

    def learn(self, batch_obs, batch_states, batch_actions, batch_logprobs, batch_advantages, batch_returns):
        # batch_obs, batch_actions, batch_logprobs are shape (total_samples, obs/act_dim)
        # batch_states, batch_returns are shape (total_samples, state/return_dim)
        
        b_obs = torch.tensor(batch_obs, dtype=torch.float32).to(self.device)
        b_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
        b_actions = torch.tensor(batch_actions, dtype=torch.int64).to(self.device)
        b_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32).to(self.device)
        b_advantages = torch.tensor(batch_advantages, dtype=torch.float32).to(self.device)
        b_returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)

        new_logprob, entropy = self.network.get_logprob_and_entropy(b_obs, b_actions)
        new_value = self.network.get_value(b_states)
        
        logratio = new_logprob - b_logprobs
        ratio = logratio.exp()

        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_value.squeeze() - b_returns) ** 2).mean()

        entropy_loss = entropy.mean()

        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()
