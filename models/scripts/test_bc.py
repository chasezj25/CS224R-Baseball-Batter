# models/scripts/test_bc.py

import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))  # Add repo root to path

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from models.envs.panda_hit_ball_env import PandaSwingBallEnv
from models.agents.bc_agent import BCPolicy

# --- Main testing function ---
def main():
    # Select device
    device_str = "mps" if torch.has_mps else "cpu"
    global device
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Register the environment
    gym.register(
        id="custom/PandaSwingBall-v0",
        entry_point="models.envs.panda_hit_ball_env:PandaSwingBallEnv",
        max_episode_steps=200,
    )

    # Create env
    env = gym.make("custom/PandaSwingBall-v0", render_mode="human")

    # Get obs/action dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Recreate the BC policy and load parameters
    policy = BCPolicy(obs_dim, action_dim, n_layers=3, hidden_dim=32)
    policy.load_state_dict(torch.load("policy_iter_0.pt", map_location=device))
    policy.to(device)
    policy.eval()

    n_episodes = 5
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0

        while True:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action.flatten())
            total_reward += reward

            if terminated or truncated:
                break

            time.sleep(0.01)  # slow down for visualization

            print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
