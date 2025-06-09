# models/scripts/run_sac.py

import sys
import os
sys.path.insert(0, os.path.abspath("."))  # Add repo root to path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from models.envs.panda_hit_ball_env import PandaSwingBallEnv

def main():
    # Register the environment
    gym.register(
        id="custom/PandaSwingBall-v0",
        entry_point="models.envs.panda_hit_ball_env:PandaSwingBallEnv",
        max_episode_steps=200,
    )

    # Create training and eval environments
    train_env = gym.make("custom/PandaSwingBall-v0")
    eval_env = gym.make("custom/PandaSwingBall-v0")

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="logs/sac_panda_hit_ball",
    )

    # Set up evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/sac_best_model",
        log_path="./logs/sac_eval_logs",
        eval_freq=50_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Train SAC model with callback
    model.learn(total_timesteps=10_000_000, callback=eval_callback)

    # Save final SAC model
    model.save("sac_panda_hit_ball")

    # Close environments
    train_env.close()
    eval_env.close()

    print("Training complete! Model saved as sac_panda_hit_ball.")

if __name__ == "__main__":
    main()