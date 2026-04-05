"""
mocap_dataset.py

PyTorch Dataset for motion capture baseball swing data.

Loads the preprocessed bat_data_100hz.pkl file (produced by preprocess/gen_bat_data.py)
and provides (observation, action) pairs suitable for behavior cloning.

Observation features per frame (16 total):
    0-2:   bat sweet spot position (x, y, z)
    3-5:   bat orientation angles (x_ang, y_ang, z_ang)
    6-8:   bat linear velocity (x_vel, y_vel, z_vel)
    9-11:  bat angular velocity (x_ang_vel, y_ang_vel, z_ang_vel)
    12-14: ball contact position (x, y, z) — fixed per swing
    15:    hit flag (0 before contact, 1 after)

Each observation stacks k=5 consecutive frames (most recent first),
giving a flattened vector of shape (k * 16,) = (80,).

Action (6 total):
    0-2: bat sweet spot position at t+1 (x, y, z)
    3-5: bat orientation angles at t+1 (x_ang, y_ang, z_ang)
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


FEATURES_PER_FRAME = 16
ACTION_DIM = 6
HISTORY_LEN = 5  # k in gen_observation


class MocapSwingDataset(Dataset):
    """
    Dataset of (observation, action) pairs extracted from motion-capture baseball swings.

    Each item is a flattened multi-frame observation paired with the next-step
    bat pose (position + orientation), enabling supervised behavior cloning.

    Args:
        data_path (str): Path to bat_data_100hz.pkl.
        split (str): 'train' or 'test'.
        train_frac (float): Fraction of swings used for training.
        seed (int): Random seed for reproducible train/test split.
        norm_stats (dict | None): Pre-computed normalization statistics
            (keys: 'obs_mean', 'obs_std', 'act_mean', 'act_std').
            If None, stats are computed from this split's data.
    """

    def __init__(self, data_path, split='train', train_frac=0.8, seed=42, norm_stats=None):
        with open(data_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Reproducible train/test split over full swings (not timesteps)
        rng = np.random.default_rng(seed)
        indices = np.arange(len(raw_data))
        rng.shuffle(indices)

        n_train = int(len(raw_data) * train_frac)
        if split == 'train':
            episode_indices = indices[:n_train]
        else:
            episode_indices = indices[n_train:]

        self.episodes = [raw_data[i] for i in episode_indices]
        self.split = split

        # Build flat (obs, action) arrays
        obs_list, act_list = [], []
        for ep in self.episodes:
            for t in range(len(ep['actions'])):
                obs_t = np.array(ep['observations'][t]).flatten()   # (k*16,)
                act_t = np.array(ep['actions'][t]).flatten()         # (6,)
                obs_list.append(obs_t)
                act_list.append(act_t)

        self.observations = np.array(obs_list, dtype=np.float32)  # (N, 80)
        self.actions = np.array(act_list, dtype=np.float32)        # (N, 6)

        # Normalization
        if norm_stats is not None:
            self.obs_mean = norm_stats['obs_mean']
            self.obs_std = norm_stats['obs_std']
            self.act_mean = norm_stats['act_mean']
            self.act_std = norm_stats['act_std']
        else:
            self.obs_mean = self.observations.mean(axis=0)
            self.obs_std = self.observations.std(axis=0) + 1e-8
            self.act_mean = self.actions.mean(axis=0)
            self.act_std = self.actions.std(axis=0) + 1e-8

        self.observations = (self.observations - self.obs_mean) / self.obs_std
        self.actions = (self.actions - self.act_mean) / self.act_std

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.observations[idx], dtype=torch.float32),
            torch.tensor(self.actions[idx], dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ob_dim(self):
        return self.observations.shape[1] if len(self.observations) > 0 else 0

    @property
    def ac_dim(self):
        return ACTION_DIM

    @property
    def norm_stats(self):
        """Return normalization statistics for sharing with other datasets."""
        return {
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
            'act_mean': self.act_mean,
            'act_std': self.act_std,
        }

    # ------------------------------------------------------------------
    # Helpers for rollout evaluation
    # ------------------------------------------------------------------

    def normalize_obs(self, obs):
        """Normalize a raw observation array (N, ob_dim) or (ob_dim,)."""
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_action(self, action_norm):
        """Map normalized network output back to original units."""
        return action_norm * self.act_std + self.act_mean

    def get_episodes_raw(self):
        """Return list of episode dicts with un-normalized data for rollout."""
        return self.episodes


def load_mocap_datasets(data_path, train_frac=0.8, seed=42):
    """
    Convenience factory that creates train and test datasets sharing
    the same normalization statistics (computed from the train split).

    Args:
        data_path (str): Path to bat_data_100hz.pkl.
        train_frac (float): Fraction of swings for training.
        seed (int): Random seed.

    Returns:
        train_dataset (MocapSwingDataset): Training set.
        test_dataset (MocapSwingDataset): Test set (using train normalization).
    """
    train_dataset = MocapSwingDataset(data_path, split='train',
                                      train_frac=train_frac, seed=seed)
    test_dataset = MocapSwingDataset(data_path, split='test',
                                     train_frac=train_frac, seed=seed,
                                     norm_stats=train_dataset.norm_stats)
    return train_dataset, test_dataset
