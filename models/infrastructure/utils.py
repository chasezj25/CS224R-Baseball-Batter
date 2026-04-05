"""
Utility functions for sampling trajectories from an environment using a policy,
and converting rollouts into a format suitable for training. Adapted from homework 1.
"""

import numpy as np
import torch

def sample_trajectory(env, policy, max_path_length):
    """
    Samples a single trajectory from the environment using the given policy.
    """
    steps = 0
    observations, actions, rewards, next_observations, terminals = [], [], [], [], []
    observation, _ = env.reset()
    
    while True:
        # Use most recent observation to get action
        observations.append(observation)
        action = policy.get_action(observation)
        action = action[0]
        actions.append(action)

        # Take a step in the environment with the action
        next_observation, reward, done, _, _ = env.step(action)
        
        # Record the results of taking the action
        steps += 1
        rewards.append(reward)
        next_observations.append(next_observation)

        # Mark the rollout as done if the episode ended or max steps reached
        rollout_done = done or steps >= max_path_length
        terminals.append(rollout_done)

        if rollout_done:
            break
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'next_observations': np.array(next_observations),
        'terminals': np.array(terminals),
    }

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length):
    """
    Samples trajectories from the environment using the given policy until the
    minimum number of timesteps is reached.
    """
    paths = []
    total_timesteps = 0

    while total_timesteps < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length)
        paths.append(path)
        total_timesteps += len(path['rewards'])

    return paths, total_timesteps

def sample_n_trajectories(env, policy, n_paths, max_path_length):
    """
    Samples a fixed number of trajectories from the environment using the given policy.
    """
    paths = []
    for _ in range(n_paths):
        path = sample_trajectory(env, policy, max_path_length)
        paths.append(path)
    
    return paths

def convert_rollouts(paths):
    """
    Utility function for adding rollouts to the replay buffer. Converts a list of paths
    into a numpy array of observations, actions, rewards, next_observations, and terminals
    suitable for training.
    """

    # Each path is a dict with keys: 'session_swing', 'observations', 'actions', 'rewards', 'terminals'
    observations = np.concatenate([np.array(path['observations']) for path in paths], axis=0)
    actions = np.concatenate([np.array(path['actions']) for path in paths], axis=0)
    rewards = np.concatenate([np.array(path['rewards']) for path in paths], axis=0)
    # If 'next_observations' is not present, infer from 'observations'
    if 'next_observations' in paths[0]:
        next_observations = np.concatenate([np.array(path['next_observations']) for path in paths], axis=0)
    else:
        next_observations = np.concatenate(
            [np.array(path['observations'])[1:] for path in paths if len(path['observations']) > 1], axis=0
        )
        # Pad last next_observation with zeros if needed
        if len(next_observations) < len(observations):
            pad_shape = observations.shape[1:]
            pad = np.zeros(pad_shape)
            next_observations = np.vstack([next_observations, pad])
    terminals = np.concatenate([np.array(path['terminals']) for path in paths], axis=0)

    return observations, actions, rewards, next_observations, terminals

def init_gpu(use_gpu=True, gpu_id=0):
    """
    Initializes the GPU for PyTorch if available.
    """
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and use_gpu:
        device = torch.device("mps")
        print("Pytorch MPS backend is available, using it.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")
    
    return device

device = init_gpu()

def from_numpy(*args, **kwargs):
    """
    Converts numpy arrays to PyTorch tensors on the initialized device.
    """
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    """
    Converts a PyTorch tensor to a numpy array.
    """
    return tensor.to('cpu').detach().numpy()