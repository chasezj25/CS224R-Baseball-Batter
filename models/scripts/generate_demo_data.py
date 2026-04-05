"""
generate_demo_data.py

Generates a synthetic bat_data_100hz.pkl-compatible dataset for testing the
behavior cloning pipeline when the real motion-capture data is unavailable.

Each synthetic swing follows a smooth, physically plausible arc (sine/cosine
curve in the horizontal plane) so that the BC agent has a learnable pattern.

Usage
-----
    python models/scripts/generate_demo_data.py \\
        --output bat_data_100hz.pkl \\
        --n_swings 200 \\
        --timesteps 150 \\
        --seed 42

The output file has the same structure as the real bat_data_100hz.pkl produced
by preprocess/gen_bat_data.py.
"""

import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

MAX_TIMESTEPS = 200
HISTORY_LEN = 5      # k frames of history per observation
FEATURES_PER_FRAME = 16
ACTION_DIM = 6

# Reward coefficients (kept consistent with gen_bat_data.py)
BAT_SPEED_COEF = 0.1
HIT_COEF = 5.0
EUCLID_DIST_COEF = -2.5


def _gen_swing(n_steps, rng, contact_frac=0.7):
    """
    Simulate one synthetic swing trajectory.

    The bat sweet spot follows a circular arc in the XY-plane, rising slightly
    along Z. Angles are derived from the tangent direction of the arc.

    Returns
    -------
    episodes : list of dicts with keys matching the real MoCap feature set.
    contact_time : float, simulated contact time (seconds).
    """
    dt = 0.01  # 100 Hz
    contact_time = n_steps * dt * contact_frac

    # Random per-swing parameters
    radius = rng.uniform(0.5, 0.9)
    phase = rng.uniform(0, 2 * math.pi)
    angular_speed = rng.uniform(3.0, 6.0)  # rad/s
    z_start = rng.uniform(0.6, 0.8)
    z_swing = rng.uniform(-0.15, 0.05)  # z displacement over swing

    # Fixed ball contact position (random within a plausible strike zone)
    ball_pos = [
        rng.uniform(-0.1, 0.1),
        rng.uniform(-0.4, -0.2),
        rng.uniform(0.55, 0.75),
    ]

    episodes = []
    for i in range(n_steps):
        t = i * dt
        theta = phase + angular_speed * t

        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        z = z_start + z_swing * (t / (n_steps * dt))

        # Velocities (finite difference from definition)
        vx = -radius * angular_speed * math.sin(theta)
        vy = radius * angular_speed * math.cos(theta)
        vz = z_swing / (n_steps * dt)

        # Orientation: angle of the bat axis in XY and XZ planes
        x_ang = math.degrees(math.atan2(vz, vy)) if (vy**2 + vz**2) > 1e-6 else 0.0
        z_ang = math.degrees(math.atan2(vy, vx)) if (vx**2 + vy**2) > 1e-6 else 0.0
        y_ang = 0.0

        # Angular velocities (approximate)
        x_ang_vel = 0.0
        y_ang_vel = 0.0
        z_ang_vel = math.degrees(angular_speed)

        if i > 0:
            prev = episodes[-1]
            x_ang_vel = (x_ang - prev["x_ang"]) / dt
            z_ang_vel = (z_ang - prev["z_ang"]) / dt

        episodes.append({
            "time": t,
            "contact_time": contact_time,
            "x": x,
            "y": y,
            "z": z,
            "x_ang": x_ang,
            "y_ang": y_ang,
            "z_ang": z_ang,
            "x_vel": vx,
            "y_vel": vy,
            "z_vel": vz,
            "x_ang_vel": x_ang_vel,
            "y_ang_vel": y_ang_vel,
            "z_ang_vel": z_ang_vel,
            "ball_pos": ball_pos,
        })

    return episodes, contact_time, ball_pos


def _gen_observation(episodes, index, ball_pos, k=HISTORY_LEN):
    """Replicate gen_observation from gen_bat_data.py."""
    ret = []
    for i in range(k):
        if index - i < 0:
            ret.append(ret[-1])
            continue
        idx = min(index - i, len(episodes) - 1)
        step = episodes[idx]
        hit = 1 if step["contact_time"] < step["time"] else 0
        vals = [
            step["x"], step["y"], step["z"],
            step["x_ang"], step["y_ang"], step["z_ang"],
            step["x_vel"], step["y_vel"], step["z_vel"],
            step["x_ang_vel"], step["y_ang_vel"], step["z_ang_vel"],
            ball_pos[0], ball_pos[1], ball_pos[2],
            hit,
        ]
        ret.append(vals)
    return ret


def generate_dataset(n_swings, max_timesteps, seed):
    """
    Generate a list of episode dicts matching the bat_data_100hz.pkl format.

    Each dict has keys: 'observations', 'next_observations', 'rewards',
    'actions', 'terminals'.
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for _ in range(n_swings):
        n_steps = rng.integers(60, max_timesteps + 1)
        episodes, contact_time, ball_pos = _gen_swing(int(n_steps), rng)

        obs_list, next_obs_list, actions, rewards, terminals = [], [], [], [], []
        ep_steps = min(MAX_TIMESTEPS, len(episodes))

        for t in range(ep_steps):
            obs_t = _gen_observation(episodes, t, ball_pos)
            obs_t1 = _gen_observation(episodes, t + 1, ball_pos)

            obs_list.append(obs_t)
            next_obs_list.append(obs_t1)

            # Action: absolute bat pose at t+1
            actions.append(obs_t1[0][:ACTION_DIM])

            # Reward matching gen_bat_data.py logic
            hit_now = obs_t[0][15]
            hit_next = obs_t1[0][15]
            reward = 0.0
            if hit_next != hit_now:
                reward += HIT_COEF
            bat_vel = math.sqrt(obs_t1[0][6]**2 + obs_t1[0][7]**2 + obs_t1[0][8]**2)
            reward += bat_vel * BAT_SPEED_COEF
            if not hit_now:
                dx = obs_t1[0][0] - obs_t1[0][12]
                dy = obs_t1[0][1] - obs_t1[0][13]
                dz = obs_t1[0][2] - obs_t1[0][14]
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                reward += EUCLID_DIST_COEF * dist
            rewards.append(reward)

            terminal = 1 if (t == ep_steps - 1) else 0
            terminals.append(terminal)

        dataset.append({
            "observations": obs_list,
            "next_observations": next_obs_list,
            "rewards": rewards,
            "actions": actions,
            "terminals": terminals,
        })

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic baseball swing data.")
    parser.add_argument('--output', '-o', type=str, default='bat_data_100hz.pkl',
                        help='Output pickle file path.')
    parser.add_argument('--n_swings', type=int, default=200,
                        help='Number of synthetic swings to generate.')
    parser.add_argument('--timesteps', type=int, default=MAX_TIMESTEPS,
                        help='Maximum timesteps per swing.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    args = parser.parse_args()

    print(f"Generating {args.n_swings} synthetic swings "
          f"(max {args.timesteps} steps each) …")

    dataset = generate_dataset(
        n_swings=args.n_swings,
        max_timesteps=args.timesteps,
        seed=args.seed,
    )

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f)

    total_steps = sum(len(ep['rewards']) for ep in dataset)
    print(f"Saved {len(dataset)} episodes ({total_steps} total timesteps) "
          f"→ {args.output}")


if __name__ == '__main__':
    main()
