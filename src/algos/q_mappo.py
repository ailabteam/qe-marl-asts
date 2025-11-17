# src/algos/q_mappo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# --- LOẠI BỎ PYQUBO ---
# from pyqubo import Array, Constraint 
from dwave.samplers import SimulatedAnnealingSampler

from src.models.actor_critic import Q_ActorCritic as ActorCritic


# --- HÀM MỚI: XÂY DỰNG QUBO BẰNG NUMPY ---
def solve_assignment_numpy(utility_matrix, sampler):
    """
    Solves the assignment problem by manually constructing the QUBO with NumPy.
    This is much more memory-efficient than using pyqubo's symbolic layer.
    """
    n_agents, n_actions = utility_matrix.shape
    
    # QUBO matrix Q
    Q = {}

    # Penalty term should be larger than the maximum possible utility difference
    P = np.max(np.abs(utility_matrix)) + 1.0

    # Objective part (linear terms)
    # The variable x_ij is mapped to index k = i * n_actions + j
    for i in range(n_agents):
        for j in range(n_actions):
            k = i * n_actions + j
            # QUBO minimizes, so we use negative utility
            Q[(k, k)] = -utility_matrix[i, j]

    # Constraint part (quadratic terms)
    # sum(x_ij for j in actions) == 1  => (sum(x_ij) - 1)^2 = 0
    # => sum(x_ij^2) - 2*sum(x_ij) + 1 = 0
    # Since x_ij is binary, x_ij^2 = x_ij.
    # => sum(x_ij) - 2*sum(x_ij) + 1 = 0 => -sum(x_ij) + 1 = 0
    # The constraint is sum(x_ij) = 1. The penalty form is P * (sum(x_ij) - 1)^2
    # P * (sum(x_ij^2) + 2*sum_{j<l}(x_ij*x_il) - 2*sum(x_ij) + 1)
    # P * (sum(x_ij) + 2*sum_{j<l}(x_ij*x_il) - 2*sum(x_ij) + 1)
    # P * (2*sum_{j<l}(x_ij*x_il) - sum(x_ij) + 1)
    for i in range(n_agents):
        # Add linear part of the constraint penalty: -P * x_ij
        for j in range(n_actions):
            k = i * n_actions + j
            if (k, k) in Q:
                Q[(k, k)] -= P
            else:
                Q[(k, k)] = -P
        
        # Add quadratic part of the constraint penalty: 2*P * x_ij * x_il
        for j in range(n_actions):
            for l in range(j + 1, n_actions):
                k1 = i * n_actions + j
                k2 = i * n_actions + l
                Q[(k1, k2)] = 2 * P

    # Solve QUBO
    sampleset = sampler.sample_qubo(Q, num_reads=1)
    solution = sampleset.first.sample

    # Decode solution
    actions = np.full(n_agents, fill_value=n_actions - 1, dtype=int) # Default to idle
    for k, v in solution.items():
        if v == 1:
            i = k // n_actions
            j = k % n_actions
            actions[i] = j
            
    return actions


class QMAPPOAgent:
    def __init__(self, n_agents, obs_shape, state_shape, n_tasks, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, vf_coef=0.5):
        # ... (phần __init__ giữ nguyên như cũ, vẫn khởi tạo sampler một lần) ...
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.action_size = n_tasks + 1
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef

        self.network = ActorCritic(obs_shape, state_shape, n_agents, n_tasks).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        
        print("Initializing QUBO Sampler (once)...")
        self.sampler = SimulatedAnnealingSampler()
        print("Sampler initialized successfully.")

    def get_joint_action(self, stacked_obs):
        # ...
        if not isinstance(stacked_obs, torch.Tensor):
            stacked_obs = torch.tensor(stacked_obs, dtype=torch.float32).to(self.device)
        if stacked_obs.ndim == 1:
            stacked_obs = stacked_obs.unsqueeze(0)
        with torch.no_grad():
            utilities = self.network.get_utilities(stacked_obs).cpu().numpy()
        if utilities.ndim == 3:
            utilities = utilities[0]
        
        # --- THAY ĐỔI LỜI GỌI HÀM ---
        joint_action = solve_assignment_numpy(utilities, self.sampler)
        return joint_action

    # ... (các hàm còn lại giữ nguyên không đổi) ...
    def get_value(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.network.get_value(state)

    def compute_advantages(self, rewards, dones, values, next_value):
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

    def learn(self, batch_stacked_obs, batch_states, batch_returns):
        b_stacked_obs = torch.tensor(batch_stacked_obs, dtype=torch.float32).to(self.device)
        b_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
        b_returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)
        
        new_values = self.network.get_value(b_states).squeeze()
        v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

        loss = self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return 0.0, v_loss.item(), 0.0
