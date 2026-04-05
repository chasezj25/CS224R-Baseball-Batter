"""
visualize_swing.py

Visualize behavior-cloning rollout trajectories against ground truth.

For each test episode the script:
  1. Runs the trained policy autoregressively (same logic as run_mocap_bc.py).
  2. Plots the predicted vs. ground-truth bat sweet-spot trajectory in 3-D.
  3. Plots per-axis position error over time.
  4. Optionally saves figures to disk.

Usage
-----
    python models/scripts/visualize_swing.py \\
        --data    bat_data_100hz.pkl \\
        --model   logs/mocap_bc/<run>/policy_best.pt \\
        --n_episodes 5 \\
        --save_dir  figures/

Requirements: matplotlib (3-D plots use mpl_toolkits.mplot3d).
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from models.agents.bc_agent import BCPolicy
from models.infrastructure.mocap_dataset import load_mocap_datasets, HISTORY_LEN, FEATURES_PER_FRAME
from models.scripts.run_mocap_bc import rollout_episode


def _load_policy(model_path, ob_dim, ac_dim, n_layers, hidden_dim, device):
    policy = BCPolicy(input_dim=ob_dim, output_dim=ac_dim,
                      n_layers=n_layers, hidden_dim=hidden_dim)
    state_dict = torch.load(model_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.network.to(device)
    policy.logstd = policy.logstd.to(device)
    policy.eval()
    return policy


def plot_episode(pred, gt, episode_idx, save_dir=None):
    """
    Generate two figures for a single episode:
        1. 3-D trajectory: predicted vs. ground-truth bat sweet spot.
        2. Per-axis position error over time.

    Args:
        pred       : np.ndarray (T, 6)  predicted actions (x,y,z,xa,ya,za).
        gt         : np.ndarray (T, 6)  ground-truth actions.
        episode_idx: int, used for figure titles / filenames.
        save_dir   : str | None, directory to save PNG files.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
    except ImportError:
        print("matplotlib not available — skipping visualisation.")
        return

    T = pred.shape[0]
    time_axis = np.arange(T) * 0.01   # seconds at 100 Hz

    # ── 3-D trajectory ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(gt[:, 0],   gt[:, 1],   gt[:, 2],
            color='steelblue', linewidth=2, label='Ground truth')
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2],
            color='tomato',    linewidth=2, linestyle='--', label='BC rollout')
    ax.scatter(*gt[0,   :3], color='green',  s=60, zorder=5, label='Start (GT)')
    ax.scatter(*gt[-1,  :3], color='navy',   s=60, zorder=5, label='End (GT)')
    ax.scatter(*pred[-1,:3], color='darkred',s=60, zorder=5, label='End (pred)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Episode {episode_idx}: Bat Sweet Spot Trajectory')
    ax.legend(fontsize=8)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f'ep{episode_idx:03d}_3d.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.join(save_dir, f'ep{episode_idx:03d}_3d.png')}")
    else:
        plt.show()

    # ── Per-axis error ────────────────────────────────────────────────────────
    labels = ['x', 'y', 'z', 'x_ang', 'y_ang', 'z_ang']
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for i, (ax, lbl) in enumerate(zip(axes.flat, labels)):
        error = pred[:, i] - gt[:, i]
        ax.plot(time_axis, error, color='tomato', linewidth=1)
        ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
        ax.set_title(f'Error: {lbl}')
        ax.set_xlabel('Time (s)')
        rmse = np.sqrt(np.mean(error**2))
        ax.set_ylabel('Pred − GT')
        ax.text(0.97, 0.95, f'RMSE={rmse:.4f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
    fig.suptitle(f'Episode {episode_idx}: Per-axis Prediction Error', y=1.01)
    plt.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(save_dir, f'ep{episode_idx:03d}_error.png'), dpi=150)
        plt.close(fig)
        print(f"  Saved: {os.path.join(save_dir, f'ep{episode_idx:03d}_error.png')}")
    else:
        plt.show()


def plot_loss_curves(logdir, save_dir=None):
    """Plot training/test loss curves if npy files are present in logdir."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    train_path = os.path.join(logdir, 'train_loss.npy')
    test_path  = os.path.join(logdir, 'test_loss.npy')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        return

    train_loss = np.load(train_path)
    test_loss  = np.load(test_path)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_loss, label='Train loss')
    ax.plot(test_loss,  label='Test loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Behavior Cloning Training Curves')
    ax.legend()
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'loss_curves.png')
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize BC rollout against ground truth MoCap swings.'
    )
    parser.add_argument('--data',        type=str, required=True,
                        help='Path to bat_data_100hz.pkl.')
    parser.add_argument('--model',       type=str, required=True,
                        help='Path to saved policy .pt weights.')
    parser.add_argument('--logdir',      type=str, default=None,
                        help='Log directory (for loss curve plots).')
    parser.add_argument('--n_episodes',  type=int, default=5,
                        help='Number of test episodes to visualise.')
    parser.add_argument('--save_dir',    type=str, default=None,
                        help='Directory to save PNG figures. '
                             'If omitted, figures are displayed interactively.')
    parser.add_argument('--num_layers',  type=int, default=3)
    parser.add_argument('--hidden_dim',  type=int, default=128)
    parser.add_argument('--train_frac',  type=float, default=0.8)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--no_gpu',      action='store_true')
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_gpu)
        else "mps" if (torch.backends.mps.is_available() and not args.no_gpu)
        else "cpu"
    )
    print(f"Device: {device}")

    # Load datasets (test split uses train normalization stats)
    _, test_ds = load_mocap_datasets(
        data_path=args.data,
        train_frac=args.train_frac,
        seed=args.seed,
    )
    print(f"Test episodes available: {len(test_ds.get_episodes_raw())}")

    policy = _load_policy(
        model_path=args.model,
        ob_dim=test_ds.ob_dim,
        ac_dim=test_ds.ac_dim,
        n_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    # Optionally plot loss curves
    if args.logdir:
        plot_loss_curves(args.logdir, save_dir=args.save_dir)

    # Rollout and visualise each episode
    episodes = test_ds.get_episodes_raw()[:args.n_episodes]
    mses = []
    for i, ep in enumerate(episodes):
        pred, gt = rollout_episode(policy, ep, test_ds.norm_stats, device)
        mse = np.mean((pred - gt) ** 2)
        mses.append(mse)
        print(f"Episode {i}: rollout MSE = {mse:.6f}  (T={len(gt)})")
        plot_episode(pred, gt, episode_idx=i, save_dir=args.save_dir)

    print(f"\nMean rollout MSE over {len(mses)} episodes: {np.mean(mses):.6f}")


if __name__ == '__main__':
    main()
