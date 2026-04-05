"""
mocap_dataset.py

PyTorch Dataset for motion capture baseball swing data.

Supports two data sources:
  1. A pre-processed bat_data_100hz.pkl file (legacy; via the default
     constructor or ``load_mocap_datasets``).
  2. Raw zip archives (landmarks.zip) read directly via
     ``MocapSwingDataset.from_raw_data()`` or ``load_mocap_datasets_raw()``.
     This path requires no intermediate pickle files.

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

    Construct with the default constructor to load from a pkl file, or use
    ``MocapSwingDataset.from_raw_data()`` to read directly from raw zip archives
    without any intermediate pickle files.

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
        self._init_from_episodes(raw_data, split, train_frac, seed, norm_stats)

    # ------------------------------------------------------------------
    # Alternative constructor: raw data pipeline (no pkl file needed)
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_data(
        cls,
        data_dir,
        split='train',
        train_frac=0.8,
        seed=42,
        norm_stats=None,
        eligible_swings_path=None,
        metadata_path=None,
        max_timesteps=200,
    ):
        """
        Build a dataset by reading raw swing data directly from the zip archives
        in *data_dir* (e.g. ``data/data/full_sig``), without loading any pkl
        file.

        Parameters
        ----------
        data_dir : str
            Directory that contains ``landmarks.zip``.
        split : str
            'train' or 'test'.
        train_frac : float
            Fraction of swings for training.
        seed : int
            Random seed.
        norm_stats : dict | None
            Pre-computed normalization statistics (pass the train set's stats
            when constructing the test set).
        eligible_swings_path : str | None
            Optional path to ``eligible_swings.csv`` to filter swings.
        metadata_path : str | None
            Optional path to ``metadata.csv`` used when *eligible_swings_path*
            is not provided (filters out left-handed swings).
        max_timesteps : int
            Maximum timesteps per episode.

        Returns
        -------
        MocapSwingDataset
        """
        import sys
        import os
        # Make sure the repo root is on the path so preprocess can be imported
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..')
        )
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        from preprocess.raw_data_pipeline import RawMocapPipeline

        pipeline = RawMocapPipeline(
            data_dir=data_dir,
            eligible_swings_path=eligible_swings_path,
            metadata_path=metadata_path,
            max_timesteps=max_timesteps,
        )
        episodes = list(pipeline.iter_episodes())

        obj = cls.__new__(cls)
        obj._init_from_episodes(episodes, split, train_frac, seed, norm_stats)
        return obj

    # ------------------------------------------------------------------
    # Shared initialisation (called by both constructors)
    # ------------------------------------------------------------------

    def _init_from_episodes(self, raw_data, split, train_frac, seed, norm_stats):
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

        obs_list, act_list = [], []
        for ep in self.episodes:
            for t in range(len(ep['actions'])):
                obs_t = np.array(ep['observations'][t]).flatten()   # (k*16,)
                act_t = np.array(ep['actions'][t]).flatten()         # (6,)
                obs_list.append(obs_t)
                act_list.append(act_t)

        self.observations = np.array(obs_list, dtype=np.float32)  # (N, 80)
        self.actions = np.array(act_list, dtype=np.float32)        # (N, 6)

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

    Loads data from a pre-processed pkl file (bat_data_100hz.pkl).
    For raw-data loading see ``load_mocap_datasets_raw``.

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


def load_mocap_datasets_raw(
    data_dir,
    train_frac=0.8,
    seed=42,
    eligible_swings_path=None,
    metadata_path=None,
    max_timesteps=200,
):
    """
    Convenience factory that creates train and test datasets directly from the
    raw zip archives — **no pkl file needed**.

    Reads landmarks.zip from *data_dir*, processes each eligible swing, and
    splits by swing (not by timestep) to avoid data leakage.  Normalization
    statistics are computed on the training split and shared with the test
    split.

    Parameters
    ----------
    data_dir : str
        Directory containing ``landmarks.zip``.
    train_frac : float
        Fraction of swings for training.
    seed : int
        Random seed.
    eligible_swings_path : str | None
        Optional path to eligible_swings.csv.
    metadata_path : str | None
        Optional path to metadata.csv.
    max_timesteps : int
        Maximum timesteps per episode.

    Returns
    -------
    train_dataset : MocapSwingDataset
    test_dataset  : MocapSwingDataset
    """
    train_dataset = MocapSwingDataset.from_raw_data(
        data_dir=data_dir,
        split='train',
        train_frac=train_frac,
        seed=seed,
        eligible_swings_path=eligible_swings_path,
        metadata_path=metadata_path,
        max_timesteps=max_timesteps,
    )
    test_dataset = MocapSwingDataset.from_raw_data(
        data_dir=data_dir,
        split='test',
        train_frac=train_frac,
        seed=seed,
        norm_stats=train_dataset.norm_stats,
        eligible_swings_path=eligible_swings_path,
        metadata_path=metadata_path,
        max_timesteps=max_timesteps,
    )
    return train_dataset, test_dataset
