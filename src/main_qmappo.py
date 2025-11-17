# src/main_mappo.py

import argparse
import os
import random
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import torch

from src.environment import SatelliteEnv
from src.algos.q_mappo import QMAPPOAgent # MODIFIED

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="qmappo", help="the name of this experiment") # MODIFIED
    # ... (các args khác giữ nguyên như main_mappo.py)
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=2048, help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--vf-coef", type=float, default=1.0, help="coefficient of the value function") # MODIFIED: vf_coef is more important now
    args = parser.parse_args()
    return args


def main(args):
    run_name = f"{args.exp_name}_seed{args.seed}_{int(time.time())}"
    print(f"Running experiment: {run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_AGENTS = 5
    N_TASKS = 20
    env = SatelliteEnv(n_agents=N_AGENTS, n_tasks=N_TASKS)

    obs_shape = env.observation_space(env.possible_agents[0]).shape
    state_shape = env.state_space().shape
    
    # MODIFIED: Initialize QMAPPOAgent
    agent = QMAPPOAgent(
        n_agents=N_AGENTS,
        obs_shape=obs_shape,
        state_shape=state_shape,
        n_tasks=N_TASKS,
        device=device,
        lr=args.learning_rate, gamma=args.gamma, gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
    )

    # --- Data Storage ---
    batch_size = args.num_steps
    minibatch_size = int(batch_size * N_AGENTS // args.num_minibatches)

    stacked_obs_buf = np.zeros((args.num_steps, N_AGENTS * obs_shape[0]), dtype=np.float32) # MODIFIED
    rewards_buf = np.zeros((args.num_steps, N_AGENTS), dtype=np.float32)
    dones_buf = np.zeros((args.num_steps, N_AGENTS), dtype=np.float32)
    values_buf = np.zeros((args.num_steps, N_AGENTS), dtype=np.float32)
    states_buf = np.zeros((args.num_steps, *state_shape), dtype=np.float32)

    # --- Training Loop ---
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = env.reset(seed=args.seed)
    next_state = env.state()
    next_done = np.zeros(N_AGENTS)
    
    log_data = []
    num_updates = args.total_timesteps // (batch_size * N_AGENTS)
    print(f"Starting Q-MAPPO training for {args.total_timesteps} total steps, which is {num_updates} updates.")

    for update in range(1, num_updates + 1):
        total_ep_reward = 0
        num_episodes = 0
        
        for step in range(0, args.num_steps):
            global_step += N_AGENTS
            
            states_buf[step] = next_state
            dones_buf[step] = next_done
            
            obs_array = np.array([next_obs[agent] for agent in env.possible_agents])
            stacked_obs_buf[step] = obs_array.flatten() # MODIFIED
            
            with torch.no_grad():
                values = agent.get_value(next_state).cpu().numpy()
            values_buf[step] = values.flatten()
            
            # MODIFIED: Get joint action from QUBO solver
            joint_action = agent.get_joint_action(obs_array.flatten())

            action_dict = {agent: act for agent, act in zip(env.possible_agents, joint_action)}
            next_obs, rewards, terminations, truncations, _ = env.step(action_dict)
            
            shared_reward = list(rewards.values())[0] if rewards else 0
            rewards_buf[step] = np.full(N_AGENTS, shared_reward)
            total_ep_reward += shared_reward
            
            if not env.agents:
                num_episodes += 1
                next_done = np.ones(N_AGENTS)
                next_obs, _ = env.reset()
            else:
                next_done = np.zeros(N_AGENTS)

            next_state = env.state()

        with torch.no_grad():
            next_value = agent.get_value(next_state).cpu().numpy().flatten()
            advantages, returns = agent.compute_advantages(rewards_buf, dones_buf, values_buf, next_value)
        
        # Flatten returns for learning
        b_returns = returns.flatten()
        # Reshape states to match the number of agents
        b_states = np.repeat(states_buf, N_AGENTS, axis=0)

        b_stacked_obs = np.repeat(stacked_obs_buf, N_AGENTS, axis=0) 


        for epoch in range(args.update_epochs):
            b_inds = np.random.permutation(batch_size * N_AGENTS)
            for start in range(0, batch_size * N_AGENTS, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                # MODIFIED: Simplified learn function call
                pg_loss, v_loss, entropy_loss = agent.learn(
                    b_stacked_obs[mb_inds], b_states[mb_inds], b_returns[mb_inds]
                )

        mean_ep_reward = total_ep_reward / num_episodes if num_episodes > 0 else float('nan')
        sps = int(global_step / (time.time() - start_time))
        
        print(f"Update {update}/{num_updates}, Step: {global_step}, SPS: {sps}")
        print(f"  Mean Episodic Reward: {mean_ep_reward:.2f}")
        print(f"  Losses -> Policy: {pg_loss:.4f}, Value: {v_loss:.4f}, Entropy: {entropy_loss:.4f}")
        print("-" * 40)
        
        log_data.append({"update": update, "global_step": global_step, "sps": sps, "mean_episodic_reward": mean_ep_reward})
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = f"{results_dir}/{args.exp_name}_seed{args.seed}.csv"
    pd.DataFrame(log_data).to_csv(csv_filename, index=False)
    print(f"Training log saved to {csv_filename}")

    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)
