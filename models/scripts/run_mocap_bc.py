"""
run_mocap_bc.py

Standalone behavior cloning trainer for motion capture baseball swing data.

Trains an MLP policy (state → next bat pose) directly on the preprocessed
bat_data_100hz.pkl file without requiring a PyBullet simulation.  The trained
model can then be used for rollout evaluation and visualization.

Data format (from preprocess/gen_bat_data.py or generate_demo_data.py)
-----------------------------------------------------------------------
Each episode is a dict:
    observations     : list of T × [k × 16] stacked-frame feature vectors
    next_observations: same, shifted by one step
    actions          : list of T × [6]  (bat position + orientation at t+1)
    rewards          : list of T floats
    terminals        : list of T ints

Observation features per frame (16 features):
    0-2:   bat sweet spot (x, y, z)
    3-5:   bat angles (x_ang, y_ang, z_ang)
    6-8:   bat linear velocity (vx, vy, vz)
    9-11:  bat angular velocity
    12-14: ball contact position (fixed per swing)
    15:    hit flag

Usage
-----
Generate demo data first (if real MoCap pkl is unavailable):

    python models/scripts/generate_demo_data.py --output bat_data_100hz.pkl

Then train:

    python models/scripts/run_mocap_bc.py \\
        --data bat_data_100hz.pkl \\
        --exp_name my_run \\
        --n_epochs 100 \\
        --batch_size 256

Outputs (written to logs/mocap_bc/<exp_name>/):
    policy.pt          : saved model weights
    train_loss.npy     : per-epoch training loss array
    test_loss.npy      : per-epoch test loss array
    rollout_mse.npy    : per-episode autoregressive rollout MSE
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from models.agents.bc_agent import BCPolicy
from models.infrastructure.mocap_dataset import load_mocap_datasets, FEATURES_PER_FRAME, HISTORY_LEN


# ──────────────────────────────────────────────────────────────────────────────
# Autoregressive rollout evaluation
# ──────────────────────────────────────────────────────────────────────────────

def rollout_episode(policy, episode, norm_stats, device):
    """
    Roll out the learned policy autoregressively on a single episode.

    Starting from the ground-truth first k frames, the policy predicts the
    next bat pose at each step, which is integrated back into the observation
    window.  The sliding window is maintained in **normalized** space to
    avoid numerical overflow from accumulated finite-difference velocities.

    Args:
        policy    : BCPolicy (eval mode).
        episode   : dict with 'observations', 'actions' lists.
        norm_stats: dict of obs/act mean and std (from MocapSwingDataset).
        device    : torch device.

    Returns:
        pred_actions : np.ndarray (T, 6) — policy predictions in original units.
        gt_actions   : np.ndarray (T, 6) — ground truth actions in original units.
    """
    obs_mean = norm_stats['obs_mean'].astype(np.float64)
    obs_std  = norm_stats['obs_std'].astype(np.float64)
    act_mean = norm_stats['act_mean'].astype(np.float64)
    act_std  = norm_stats['act_std'].astype(np.float64)

    feat = FEATURES_PER_FRAME
    k    = HISTORY_LEN
    dt   = 0.01  # 100 Hz

    raw_obs  = [np.array(o, dtype=np.float64).flatten() for o in episode['observations']]
    raw_acts = [np.array(a, dtype=np.float64).flatten() for a in episode['actions']]
    T = len(raw_acts)

    # Per-feature normalization stats for each window slot (slot 0 = most-recent)
    # obs_mean/std layout: [frame0_feat0..15, frame1_feat0..15, ...]
    slot_mean = [obs_mean[i * feat:(i + 1) * feat] for i in range(k)]
    slot_std  = [obs_std[ i * feat:(i + 1) * feat] for i in range(k)]

    def _norm_frame(raw_f, slot):
        return (raw_f - slot_mean[slot]) / slot_std[slot]

    # Initialise sliding window from ground-truth frames; kept in normalized space
    raw_window  = [raw_obs[0][i * feat:(i + 1) * feat].copy() for i in range(k)]
    norm_window = [_norm_frame(raw_window[i], i) for i in range(k)]

    pred_actions, gt_actions = [], []

    policy.eval()
    with torch.no_grad():
        for t in range(T):
            obs_norm   = np.concatenate(norm_window).astype(np.float32)
            obs_tensor = torch.tensor(obs_norm).unsqueeze(0).to(device)

            dist     = policy(obs_tensor)
            act_norm = dist.mean.squeeze(0).cpu().numpy().astype(np.float64)
            act_pred = act_norm * act_std + act_mean   # denormalize → original units

            pred_actions.append(act_pred.copy())
            gt_actions.append(raw_acts[t].copy())

            # Build the new raw frame from the predicted bat pose
            prev_raw = raw_window[0]
            new_raw  = prev_raw.copy()
            new_raw[0:6] = act_pred[:6]  # position + angles

            # Finite-difference velocities, clipped to prevent blow-up
            vel = (act_pred[:6] - prev_raw[:6]) / dt
            new_raw[6:12] = np.clip(vel, -1e4, 1e4)

            # Ball pos and hit flag: propagate from ground truth when available
            if t + 1 < len(raw_obs):
                new_raw[12:16] = raw_obs[t + 1][12:16]

            # Slide window; re-normalize each slot for its position's stats
            raw_window  = [new_raw] + raw_window[:-1]
            norm_window = [_norm_frame(raw_window[i], i) for i in range(k)]

    return np.array(pred_actions), np.array(gt_actions)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(params):
    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not params['no_gpu'])
        else "mps" if (torch.backends.mps.is_available() and not params['no_gpu'])
        else "cpu"
    )
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading MoCap dataset …")
    train_ds, test_ds = load_mocap_datasets(
        data_path=params['data'],
        train_frac=params['train_frac'],
        seed=params['seed'],
    )
    print(f"  Train: {len(train_ds)} samples | Test: {len(test_ds)} samples")
    print(f"  ob_dim={train_ds.ob_dim}  ac_dim={train_ds.ac_dim}")

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'],
                              shuffle=True, drop_last=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=params['batch_size'],
                              shuffle=False, drop_last=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    policy = BCPolicy(
        input_dim=train_ds.ob_dim,
        output_dim=train_ds.ac_dim,
        n_layers=params['num_layers'],
        hidden_dim=params['hidden_dim'],
        learning_rate=params['learning_rate'],
    )
    policy.network.to(device)
    policy.logstd = policy.logstd.to(device)

    loss_fn = nn.MSELoss()

    # ── Logging dir ───────────────────────────────────────────────────────────
    logdir = params['logdir']
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging to: {logdir}")

    train_losses, test_losses = [], []
    best_test_loss = float('inf')

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Starting training …")
    for epoch in range(1, params['n_epochs'] + 1):
        policy.network.train()
        epoch_loss = 0.0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)

            dist = policy(obs_batch)
            pred = dist.mean               # use mean for deterministic loss
            loss = loss_fn(pred, act_batch)

            policy.optimizer.zero_grad()
            loss.backward()
            policy.optimizer.step()
            epoch_loss += loss.item() * len(obs_batch)

        epoch_loss /= len(train_ds)
        train_losses.append(epoch_loss)

        # ── Validation ────────────────────────────────────────────────────────
        policy.network.eval()
        with torch.no_grad():
            test_loss = sum(
                loss_fn(policy(ob.to(device)).mean, ac.to(device)).item() * len(ob)
                for ob, ac in test_loader
            ) / max(len(test_ds), 1)
        test_losses.append(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(policy.state_dict(), os.path.join(logdir, 'policy_best.pt'))

        if epoch % max(1, params['n_epochs'] // 10) == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{params['n_epochs']} "
                  f"| train_loss={epoch_loss:.6f} "
                  f"| test_loss={test_loss:.6f}")

    # ── Save final model and loss curves ──────────────────────────────────────
    torch.save(policy.state_dict(), os.path.join(logdir, 'policy.pt'))
    np.save(os.path.join(logdir, 'train_loss.npy'), np.array(train_losses))
    np.save(os.path.join(logdir, 'test_loss.npy'),  np.array(test_losses))
    print(f"Model saved → {logdir}/policy.pt  (best: {best_test_loss:.6f})")

    # ── Autoregressive rollout evaluation ─────────────────────────────────────
    print("\nEvaluating autoregressive rollout on test episodes …")
    policy.load_state_dict(torch.load(os.path.join(logdir, 'policy_best.pt'),
                                       map_location=device))
    policy.network.eval()

    rollout_mses = []
    test_episodes = test_ds.get_episodes_raw()[:params['n_eval_episodes']]
    for ep in test_episodes:
        pred, gt = rollout_episode(policy, ep, test_ds.norm_stats, device)
        mse = np.mean((pred - gt) ** 2)
        rollout_mses.append(mse)

    mean_rollout_mse = np.mean(rollout_mses) if rollout_mses else float('nan')
    np.save(os.path.join(logdir, 'rollout_mse.npy'), np.array(rollout_mses))
    print(f"  Rollout MSE (mean over {len(rollout_mses)} episodes): "
          f"{mean_rollout_mse:.6f}")

    print("\nDone.")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a behavior-cloning policy on MoCap baseball swing data."
    )
    parser.add_argument('--data', '-d', type=str, required=True,
                        help='Path to bat_data_100hz.pkl.')
    parser.add_argument('--exp_name', type=str, default='mocap_bc',
                        help='Experiment name for the log directory.')

    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of hidden layers in the MLP.')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer width.')

    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of swings for training.')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--n_eval_episodes', type=int, default=20,
                        help='Number of test episodes for rollout evaluation.')
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()
    params = vars(args)

    # Build log directory
    base = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        '../../logs/mocap_bc')
    logdir = os.path.join(base,
                          f"{args.exp_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    params['logdir'] = logdir

    train(params)


if __name__ == '__main__':
    main()
