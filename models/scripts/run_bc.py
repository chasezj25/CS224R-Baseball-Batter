import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from agents import BCAgent
from preprocess import load_trajectories
from swing_env import SwingEnv
import os
import numpy as np

# Converts a list of reference trajectories into BC-style training data
# Each observation is a vector of joint angles from one frame,
# and the corresponding action is the joint angles from the next frame
def make_bc_dataset(trajectories):
    obs_list, act_list = [], []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            curr_frame = traj[i]
            next_frame = traj[i + 1]

            obs = []  # Current frame joint angles
            act = []  # Next frame joint angles
            for key in curr_frame:
                if "angle" in key:  # Only use joint angle values
                    obs.append(curr_frame[key])
                    act.append(next_frame[key])
            obs_list.append(obs)
            act_list.append(act)

    obs_tensor = torch.tensor(obs_list, dtype=torch.float32)
    act_tensor = torch.tensor(act_list, dtype=torch.float32)
    return obs_tensor, act_tensor

# Trains the agent using MSE loss between predicted and target next frame joint angles
def train(agent, obs_tensor, act_tensor, device, batch_size):
    agent.model.train()
    total_loss = 0
    num_batches = 0
    for i in range(0, len(obs_tensor), batch_size):
        batch_obs = obs_tensor[i:i+batch_size].to(device)  # Input: current state
        batch_act = act_tensor[i:i+batch_size].to(device)  # Target: next state
        loss = agent.update({'obs': batch_obs, 'actions': batch_act.unsqueeze(1)}, device)
        total_loss += loss
        num_batches += 1
        if num_batches % 10 == 0:
            print(f"  Train batch {num_batches}: batch loss = {loss:.4f}")
    return total_loss / num_batches

# Evaluates the agent without gradient updates, printing batch loss occasionally
def evaluate(agent, obs_tensor, act_tensor, device, batch_size):
    agent.model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i in range(0, len(obs_tensor), batch_size):
            batch_obs = obs_tensor[i:i+batch_size].to(device)
            batch_act = act_tensor[i:i+batch_size].to(device)
            loss = agent.evaluate({'obs': batch_obs, 'actions': batch_act.unsqueeze(1)}, device)
            total_loss += loss
            num_batches += 1
            if num_batches % 10 == 0:
                print(f"  Eval batch {num_batches}: batch loss = {loss:.4f}")
    return total_loss / num_batches

# Performs a rollout in the SwingEnv using the current policy.
# Returns the average squared error between agent state and reference over the rollout
def rollout(agent, env, device, steps=100):
    obs = env.reset()  # Initial environment state
    total_error = 0.0
    for t in range(steps):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = agent.model(obs_tensor).squeeze(0).cpu().numpy()  # Predict next action (joint angles)
        obs, _, done, _ = env.step(action)  # Apply action in the environment
        ref = env.get_reference_frame()  # Ground truth for current timestep
        error = np.mean((obs - ref) ** 2)  # MSE between predicted and reference
        total_error += error
        if done:
            break
    return total_error / (t + 1)

# Main training loop: loads data, initializes model/env, trains agent, logs results
def main(args):
    writer = SummaryWriter(log_dir=args.log_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading trajectories from:", args.data_path)
    trajectories = load_trajectories(args.data_path)  # Load CSV motion capture data
    obs_tensor, act_tensor = make_bc_dataset(trajectories)  # Convert to BC format

    # Split into train and validation sets
    train_size = int(0.8 * len(obs_tensor))
    train_obs, val_obs = obs_tensor[:train_size], obs_tensor[train_size:]
    train_act, val_act = act_tensor[:train_size], act_tensor[train_size:]

    print(f"Dataset size: {len(obs_tensor)} samples")
    print(f"Train set: {len(train_obs)} samples | Validation set: {len(val_obs)} samples")

    # Initialize BC agent
    agent = BCAgent(obs_dim=train_obs.shape[1], action_dim=train_act.shape[1]).to(device)
    os.makedirs(args.model_dir, exist_ok=True)

    # Initialize environment with reference trajectories
    env = SwingEnv(reference_trajectories=trajectories)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss = train(agent, train_obs, train_act, device, args.batch_size)
        val_loss = evaluate(agent, val_obs, val_act, device, args.batch_size)
        rollout_error = rollout(agent, env, device)  # Test rollout using trained model

        # Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Rollout/MSE", rollout_error, epoch)

        # Print summary every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Rollout MSE: {rollout_error:.4f}")

        # Save checkpoints periodically
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            model_path = os.path.join(args.model_dir, f"bc_model_epoch{epoch+1}.pt")
            torch.save(agent.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")

    # Save final model
    final_model_path = os.path.join(args.model_dir, "bc_model_final.pt")
    torch.save(agent.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to swing dataset CSV')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model_dir', type=str, default="checkpoints")
    parser.add_argument('--log_dir', type=str, default="runs")
    parser.add_argument('--save_every', type=int, default=5, help='Save model every n epochs')
    args = parser.parse_args()
    main(args)