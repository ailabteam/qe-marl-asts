# src/main.py

import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import pandas as pd
import torch

from src.environment import SatelliteEnv
from src.algos.ppo import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="ppo_single_agent",
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-steps", type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    
    args = parser.parse_args()
    return args

def main(args):
    run_name = f"{args.exp_name}_seed{args.seed}_{int(time.time())}"
    print(f"Running experiment: {run_name}")

    # --- 1. Seeding ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # --- 2. Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = SatelliteEnv(n_agents=3, n_tasks=10)

    agent_to_train = "satellite_0"
    obs_shape = env.observation_space(agent_to_train).shape
    action_size = env.action_space(agent_to_train).n
    
    agent = PPOAgent(
        obs_shape=obs_shape,
        action_size=action_size,
        device=device,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
    )

    # --- 3. Data Storage ---
    batch_size = args.num_steps
    minibatch_size = int(batch_size // args.num_minibatches)

    obs = np.zeros((args.num_steps, *obs_shape), dtype=np.float32)
    actions = np.zeros((args.num_steps,), dtype=np.int64)
    logprobs = np.zeros((args.num_steps,), dtype=np.float32)
    rewards = np.zeros((args.num_steps,), dtype=np.float32)
    dones = np.zeros((args.num_steps,), dtype=np.float32)
    values = np.zeros((args.num_steps,), dtype=np.float32)

    # --- 4. Training Loop ---
    global_step = 0
    start_time = time.time()
    
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = next_obs[agent_to_train]
    next_done = 0
    
    log_data = []
    num_updates = args.total_timesteps // batch_size
    print(f"Starting training for {args.total_timesteps} timesteps, which is {num_updates} updates.")

    for update in range(1, num_updates + 1):
        ep_rewards = [] # Track rewards for this rollout
        for step in range(0, args.num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.cpu().numpy()
            actions[step] = action.cpu().numpy()
            logprobs[step] = logprob.cpu().numpy()

            full_actions = {ag: env.action_space(ag).sample() for ag in env.agents}
            full_actions[agent_to_train] = action.cpu().numpy().item()
            
            next_obs_dict, rewards_dict, terminations, truncations, _ = env.step(full_actions)
            
            current_reward = rewards_dict.get(agent_to_train, 0)
            rewards[step] = current_reward
            ep_rewards.append(current_reward)

            if not env.agents:
                next_done = 1
                next_obs, _ = env.reset()
                next_obs = next_obs[agent_to_train]
            else:
                next_done = 0
                next_obs = next_obs_dict[agent_to_train]
        
        with torch.no_grad():
            next_value = agent.get_action_and_value(next_obs)[3].cpu().numpy()
            advantages, returns = agent.compute_advantages(rewards, dones, values, next_value)
        
        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        for epoch in range(args.update_epochs):
            b_inds = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]
                
                pg_loss, v_loss, entropy_loss = agent.learn(
                    b_obs[mb_inds], b_actions[mb_inds], b_logprobs[mb_inds],
                    b_advantages[mb_inds], b_returns[mb_inds]
                )

        # --- Logging ---
        mean_ep_reward = np.sum(ep_rewards) / np.sum(dones) if np.sum(dones) > 0 else float('nan')
        sps = int(global_step / (time.time() - start_time))
        
        print(f"Update {update}/{num_updates}, Step: {global_step}, SPS: {sps}")
        print(f"  Mean Episodic Reward: {mean_ep_reward:.2f}")
        print(f"  Losses -> Policy: {pg_loss:.4f}, Value: {v_loss:.4f}, Entropy: {entropy_loss:.4f}")
        print("-" * 40)
        
        log_data.append({
            "update": update,
            "global_step": global_step,
            "sps": sps,
            "mean_episodic_reward": mean_ep_reward,
            "policy_loss": pg_loss,
            "value_loss": v_loss,
            "entropy_loss": entropy_loss,
        })
    
    # --- 5. Save Results ---
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Use a more descriptive filename
    csv_filename = f"{results_dir}/{args.exp_name}_seed{args.seed}.csv"
    
    df = pd.DataFrame(log_data)
    df.to_csv(csv_filename, index=False)
    print(f"Training log saved to {csv_filename}")

    env.close()
    print("Training finished.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
