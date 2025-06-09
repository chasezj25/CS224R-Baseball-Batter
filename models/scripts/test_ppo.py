import sys
import os
import time
sys.path.insert(0, os.path.abspath("."))  # Add repo root to path

import gymnasium as gym
from stable_baselines3 import PPO
from models.envs.panda_hit_ball_env import PandaSwingBallEnv

def main():
    # Register the environment (you need this even when testing)
    gym.register(
        id="custom/PandaSwingBall-v0",
        entry_point="models.envs.panda_hit_ball_env:PandaSwingBallEnv",
        max_episode_steps=200,
    )

    # Load the trained model
    model = PPO.load("ppo_panda_hit_ball")

    # Create a single env for visualization
    env = gym.make("custom/PandaSwingBall-v0", render_mode="human")  # set render_mode to "human" if your env supports it

    n_episodes = 5
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # Use deterministic=True for evaluation
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            time.sleep(0.02)  # Slow down rendering if needed

        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()