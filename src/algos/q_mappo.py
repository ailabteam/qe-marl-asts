# src/algos/q_mappo.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyqubo import Array, Constraint
from dwave.samplers import SimulatedAnnealingSampler

# Đổi tên class import để tránh trùng lặp
from src.models.actor_critic import Q_ActorCritic as ActorCritic

def solve_assignment_qubo(utility_matrix):
    """
    Solves the assignment problem using QUBO for a single utility matrix.
    
    Args:
        utility_matrix (np.ndarray): A 2D numpy array of shape (n_agents, n_actions).
    
    Returns:
        np.ndarray: An array of actions of shape (n_agents,).
    """
    n_agents, n_actions = utility_matrix.shape
    
    # Create binary variables x_ij = 1 if agent i is assigned action j
    x = Array.create('x', shape=(n_agents, n_actions), vartype='BINARY')
    
    # Objective function: Maximize total utility
    # QUBO minimizes, so we minimize the negative utility
    objective = -np.sum(utility_matrix * x)
    
    # Constraint 1: Each agent must be assigned exactly one action
    # sum(x_ij for j in actions) == 1 for each agent i
    constraints = [Constraint((np.sum(x[i, :]) - 1)**2, label=f'agent_{i}_constraint') for i in range(n_agents)]
    
    # Compile the model. Penalty chosen to be larger than max possible utility.
    # Add a small epsilon to avoid P=0 if all utilities are 0.
    P = np.max(np.abs(utility_matrix)) + 1.0 
    model = (objective + P * np.sum(constraints)).compile()
    
    # Get the QUBO formulation
    qubo, offset = model.to_qubo()
    
    # Solve using Simulated Annealing. Using a singleton sampler for efficiency.
    # Note: In a real high-performance setting, you might initialize the sampler once.
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo, num_reads=1)
    
    # Decode the best solution
    decoded_solution = model.decode_sample(sampleset.first.sample, vartype='BINARY')
    
    # Extract actions
    actions = np.zeros(n_agents, dtype=int)
    for i in range(n_agents):
        action_found = False
        for j in range(n_actions):
            if decoded_solution.array('x', (i, j)) == 1:
                actions[i] = j
                action_found = True
                break
        # If solver fails to assign an action due to conflicts (rare), assign a default action (e.g., idle)
        if not action_found:
            actions[i] = n_actions - 1 # Assuming last action is 'idle'
            
    return actions

class QMAPPOAgent:
    def __init__(self, n_agents, obs_shape, state_shape, n_tasks, device, lr=3e-4, gamma=0.99, gae_lambda=0.95, vf_coef=0.5):
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.action_size = n_tasks + 1
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef

        self.network = ActorCritic(obs_shape, state_shape, n_agents, n_tasks).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def get_joint_action(self, stacked_obs):
        """
        Get a joint action for all agents using the QUBO solver.
        """
        if not isinstance(stacked_obs, torch.Tensor):
            stacked_obs = torch.tensor(stacked_obs, dtype=torch.float32).to(self.device)
        
        # Ensure the input has a batch dimension
        if stacked_obs.ndim == 1:
            stacked_obs = stacked_obs.unsqueeze(0)

        with torch.no_grad():
            utilities = self.network.get_utilities(stacked_obs).cpu().numpy()
        
        # The network outputs a batch, we only need the first element
        if utilities.ndim == 3:
            utilities = utilities[0]
        
        # Solve QUBO to get the joint action
        joint_action = solve_assignment_qubo(utilities)
        return joint_action

    def get_value(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
        # Ensure the input has a batch dimension
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self.network.get_value(state)

    def compute_advantages(self, rewards, dones, values, next_value):
        # Same as MAPPO
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
        """
        Learning in Q-MAPPO (simplified version). 
        The actor is trained implicitly through the critic's gradient.
        We only explicitly compute and optimize the value loss.
        """
        b_stacked_obs = torch.tensor(batch_stacked_obs, dtype=torch.float32).to(self.device)
        b_states = torch.tensor(batch_states, dtype=torch.float32).to(self.device)
        b_returns = torch.tensor(batch_returns, dtype=torch.float32).to(self.device)
        
        # Critic loss
        new_values = self.network.get_value(b_states).squeeze()
        v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()

        # In this simplified model, the actor's learning is driven by the critic's loss.
        # The total loss is just the value loss.
        loss = self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), 0.5) # Gradient clipping
        self.optimizer.step()

        # Return 0 for policy and entropy loss as they are not explicitly calculated
        return 0.0, v_loss.item(), 0.0
