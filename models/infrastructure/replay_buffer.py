"""
Replay Buffer for storing and sampling experiences in reinforcement learning.
Adapted from the original code in homework 1.
"""
from models.infrastructure.utils import *
import numpy as np

class ReplayBuffer():
    """
    ReplayBuffer is a data structure for storing and sampling experience rollouts in reinforcement learning.
    It efficiently manages a fixed-size buffer of transitions for training RL agents.
    """
    def __init__(self, max_size=1000000):
        # Storage for each rollout
        self.paths = []
        self.max_size = max_size

        # Store component arrays from each rollout
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None

    def __len__(self):
        """
        Returns the number of transitions stored in the replay buffer.
        """
        if self.observations is not None:
            return self.observations.shape[0]
        else:
            return 0
        
    def add_rollouts(self, paths):
        """
        Add new rollout trajectories to the replay buffer.
        """
        # Add new rollouts to the buffer
        for path in paths:
            self.paths.append(path)

        observations, actions, rewards, next_observations, terminals = convert_rollouts(paths)

        # Add the components to their respective arrays
        if self.observations is None:
            self.observations = observations[-self.max_size:]
            self.actions = actions[-self.max_size:]
            self.rewards = rewards[-self.max_size:]
            self.next_observations = next_observations[-self.max_size:]
            self.terminals = terminals[-self.max_size:]
        else:
            self.observations = np.concatenate((self.observations, observations))[-self.max_size:]
            self.actions = np.concatenate((self.actions, actions))[-self.max_size:]
            self.rewards = np.concatenate((self.rewards, rewards))[-self.max_size:]
            self.next_observations = np.concatenate((self.next_observations, next_observations))[-self.max_size:]
            self.terminals = np.concatenate((self.terminals, terminals))[-self.max_size:]

    def sample_random_data(self, batch_size):
        """
        Samples a random batch of data from the replay buffer.
        """
        indices = np.random.permutation(self.actions.shape[0])[:batch_size]

        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.terminals[indices]
        )
    
    def sample_recent_data(self, batch_size=1):
        """
        Samples the most recent batch of data from the replay buffer.
        """
        return (
            self.observations[-batch_size:],
            self.actions[-batch_size:],
            self.rewards[-batch_size:],
            self.next_observations[-batch_size:],
            self.terminals[-batch_size:]
        )


