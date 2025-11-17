# src/environment.py

import functools
from copy import copy

import numpy as np
from gymnasium.spaces import Box, Discrete

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.env import AgentID, ObsType, ActionType

# --- Helper Functions ---
def get_distance(pos1, pos2):
    """Calculates Euclidean distance between two points."""
    return np.linalg.norm(pos1 - pos2)

# --- Environment Definition ---
class SatelliteEnv(ParallelEnv):
    metadata = {
        "name": "satellite_v0",
        "is_parallelizable": True,
    }

    def __init__(self, n_agents=10, n_tasks=50, map_size=100, render_mode=None):
        """
        n_agents: number of satellites
        n_tasks: number of tasks that appear over an episode
        map_size: size of the 2D map (from -map_size to +map_size)
        """
        super().__init__()
        self.n_agents = n_agents
        self.n_tasks = n_tasks
        self.map_size = map_size
        self.render_mode = render_mode

        # Define agents
        self.possible_agents = [f"satellite_{i}" for i in range(self.n_agents)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # Internal state
        self.agents = copy(self.possible_agents)
        self.episode_step = 0
        self.max_steps = 200 # An episode ends after this many steps

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        # Observation: [my_x, my_y, task1_x, task1_y, task1_prio, ..., taskK_x, taskK_y, taskK_prio]
        # We assume a maximum of `n_tasks` can be observed at once.
        obs_size = 2 + self.n_tasks * 3
        return Box(low=-self.map_size, high=self.map_size, shape=(obs_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        # Action: Choose which task to observe (from 0 to n_tasks-1) or do nothing (action n_tasks)
        return Discrete(self.n_tasks + 1)
    
    @functools.lru_cache(maxsize=None)
    def state_space(self):
        """Returns the shape of the global state."""
        # Global state: [all_sat_pos, all_task_info]
        # (N * 2 for satellite positions, M * 3 for task info)
        state_size = self.n_agents * 2 + self.n_tasks * 3
        return Box(low=-self.map_size, high=self.map_size, shape=(state_size,), dtype=np.float32)

    def state(self) -> np.ndarray:
        """Returns the global state of the environment."""
        state_vector = np.zeros(self.state_space().shape, dtype=np.float32)
        
        # Satellite positions
        state_vector[0 : self.n_agents * 2] = self.satellite_positions.flatten()
        
        # Task information
        task_start_idx = self.n_agents * 2
        for i, task_id in enumerate(self.task_list):
            task = self.tasks[task_id]
            if not task["completed"]:
                state_vector[task_start_idx + i*3 : task_start_idx + (i+1)*3] = [
                    task["position"][0],
                    task["position"][1],
                    task["priority"],
                ]
        return state_vector


    def reset(self, seed=None, options=None):
        self.agents = copy(self.possible_agents)
        self.episode_step = 0

        # Initialize satellite positions (randomly on a circle for simplicity)
        self.satellite_positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, self.n_agents, endpoint=False)
        radius = self.map_size * 0.8
        self.satellite_positions[:, 0] = radius * np.cos(angles)
        self.satellite_positions[:, 1] = radius * np.sin(angles)
        self.satellite_velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        # Simple circular motion
        self.satellite_velocities[:, 0] = -radius * np.sin(angles) * 0.05
        self.satellite_velocities[:, 1] = radius * np.cos(angles) * 0.05


        # Initialize tasks
        self.tasks = {
            f"task_{i}": {
                "position": np.random.uniform(-self.map_size, self.map_size, 2).astype(np.float32),
                "priority": np.random.randint(1, 11),
                "completed": False,
            }
            for i in range(self.n_tasks)
        }
        self.task_list = list(self.tasks.keys()) # Keep a consistent order

        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # Thay thế hàm _get_obs cũ bằng hàm này
    def _get_obs(self, agent: AgentID):
        agent_idx = self.agent_name_mapping[agent]
        my_pos = self.satellite_positions[agent_idx]

        # Sửa dòng dưới đây: gọi self.observation_space(agent) như một hàm
        obs = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
        obs[0:2] = my_pos

        # Add task info
        # For simplicity, all tasks are visible to all agents for now
        # In a real scenario, this would be based on sensor range
        task_info_start_idx = 2
        for i, task_id in enumerate(self.task_list):
            task = self.tasks[task_id]
            if not task["completed"]:
                obs[task_info_start_idx + i*3 : task_info_start_idx + (i+1)*3] = [
                    task["position"][0],
                    task["position"][1],
                    task["priority"],
                ]
        return obs

    def step(self, actions: dict[AgentID, ActionType]):
        self.episode_step += 1
        rewards = {agent: 0.0 for agent in self.agents}

        # 1. Process actions
        completed_tasks_this_step = set()
        for agent, action in actions.items():
            # Action is an integer from 0 to n_tasks
            if action < self.n_tasks:
                task_id = self.task_list[action]
                if not self.tasks[task_id]["completed"]:
                    # Simple completion logic: if chosen, task is done
                    # A more complex model could check for distance
                    completed_tasks_this_step.add(task_id)

        # Calculate shared reward
        shared_reward = 0
        for task_id in completed_tasks_this_step:
            if not self.tasks[task_id]["completed"]:
                shared_reward += self.tasks[task_id]["priority"]
                self.tasks[task_id]["completed"] = True

        for agent in self.agents:
            rewards[agent] = shared_reward

        # 2. Update environment state (move satellites)
        self.satellite_positions += self.satellite_velocities

        # 3. Check for termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.episode_step >= self.max_steps for agent in self.agents}

        # 4. Get next observations and infos
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if any(truncations.values()):
            self.agents = [] # End of episode

        return observations, rewards, terminations, truncations, infos

    def close(self):
        pass
