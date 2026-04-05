# CS224R-Baseball-Batter
Stanford CS 224R Final Project Spring 2025. A RL policy for a simulated robotic agent capable of hitting baseball.

Team:
Chase Joyner
Mack Smith

---

## Project Overview

This project develops a **behavior cloning (BC)** system that imitates baseball swing motion from motion-capture (MoCap) data, alongside reinforcement learning baselines (PPO/SAC) in a PyBullet Panda-arm simulation.

### Repository Structure

```
models/
  agents/        # BCAgent, BCPolicy (MLP)
  envs/          # PandaSwingBallEnv (PyBullet simulation)
  infrastructure/
    bc_trainer.py     # RL-env-based BC trainer (uses gym environment)
    mocap_dataset.py  # PyTorch Dataset (pkl OR raw zip — no pkl required)
    replay_buffer.py  # Experience replay buffer
    utils.py          # Sampling utilities
  scripts/
    generate_demo_data.py  # Synthetic swing generator for testing
    run_mocap_bc.py        # MoCap BC training + evaluation (pkl or raw)
    visualize_swing.py     # Rollout visualization
    run_bc.py              # Env-based BC trainer entry point
    run_ppo.py             # PPO training
    run_sac.py             # SAC training
preprocess/
  raw_data_pipeline.py   # Streams episodes from landmarks.zip (no pkl needed)
  gen_sim_expert_data.py # Raw MoCap -> PyBullet IK -> expert_sim_data.pkl
  gen_bat_data.py        # MoCap CSV -> bat_data_100hz.pkl (legacy)
  filter_data.py         # Filter left-handed swings
  sort_data.py           # Sort / organize raw data
```

---

## Behavior Cloning from Motion Capture

### Working directly with raw data (no pkl file needed)

The entire pipeline now works directly from the raw zip archives in
`data/data/full_sig/` — **no intermediate pickle files are required**.

#### Raw data directory layout

```
data/data/full_sig/
  landmarks.zip       <- bat + body landmark positions (streamed directly)
  joint_angles.zip    <- joint angles (used by legacy sort_data.py)
  joint_velos.zip     <- joint velocities (used by legacy sort_data.py)
```

#### Generating simulation expert data from raw MoCap

```bash
# Produces expert_sim_data.pkl with observations/actions in the Panda sim's
# coordinate space (14-dim obs, 7-dim joint-delta actions) via PyBullet IK.
python preprocess/gen_sim_expert_data.py \
    --data_dir data/data/full_sig \
    --output expert_sim_data.pkl \
    --urdf models/envs/panda_arm_bat.urdf

# Optionally limit to right-handed swings via metadata filter:
python preprocess/gen_sim_expert_data.py \
    --data_dir  data/data/full_sig \
    --eligible  eligible_swings.csv \
    --output    expert_sim_data.pkl
```

The output is immediately usable with the simulation BC trainer:
```bash
python models/scripts/run_bc.py \
    --expert_data expert_sim_data.pkl \
    --exp_name sim_bc_from_raw
```

#### Training the MoCap BC policy (bat-pose imitation) from raw data

```bash
# Read directly from landmarks.zip — no bat_data_100hz.pkl needed:
python models/scripts/run_mocap_bc.py \
    --raw_data_dir data/data/full_sig \
    --exp_name raw_bc_run \
    --n_epochs 200 \
    --batch_size 256
```

### Data format

The preprocessing pipeline (`preprocess/gen_bat_data.py`) converts raw MoCap CSV data into `bat_data_100hz.pkl`, a list of episode dicts:

```
observations     : list of T x [k=5 x 16] stacked feature frames
next_observations: same, shifted by one step
actions          : list of T x [6]  (bat position + orientation at t+1)
rewards          : list of T floats
terminals        : list of T ints (0 or 1)
```

**Observation features (16 per frame, 5 frames stacked -> 80-dim vector):**

| Index | Feature |
|-------|---------|
| 0-2   | Bat sweet-spot position (x, y, z) |
| 3-5   | Bat orientation angles (x_ang, y_ang, z_ang) |
| 6-8   | Bat linear velocity (vx, vy, vz) |
| 9-11  | Bat angular velocity |
| 12-14 | Ball contact position (fixed per swing) |
| 15    | Hit flag (0 before contact, 1 after) |

**Action (6-dim):** bat sweet-spot position + orientation at the next timestep.

### Quick start

**Step 1 — Prepare data**

*Option A: Work directly from raw data (recommended — no pkl required)*
```bash
# The landmarks.zip archive is read in-place; nothing to extract.
# Pass --raw_data_dir directly to run_mocap_bc.py or gen_sim_expert_data.py.
```

*Option B: Generate legacy pkl from raw MoCap CSVs*
```bash
cd preprocess
python filter_data.py      # filter left-handed swings
python gen_bat_data.py     # produce bat_data_100hz.pkl
```

*Option C: Synthetic demo data (no MoCap files needed)*
```bash
python models/scripts/generate_demo_data.py \
    --output bat_data_100hz.pkl \
    --n_swings 200 \
    --seed 42
```

**Step 2 — Train the BC policy**
```bash
# Raw data (no pkl needed):
python models/scripts/run_mocap_bc.py \
    --raw_data_dir data/data/full_sig \
    --exp_name my_experiment \
    --n_epochs 200 \
    --batch_size 256

# Or from a pre-built pkl:
python models/scripts/run_mocap_bc.py \
    --data bat_data_100hz.pkl \
    --exp_name my_experiment \
    --n_epochs 200 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --num_layers 3 \
    --hidden_dim 128
```
Results are saved under `logs/mocap_bc/<exp_name>_<timestamp>/`:
- `policy.pt` / `policy_best.pt` - model weights
- `train_loss.npy` / `test_loss.npy` - per-epoch loss arrays
- `rollout_mse.npy` - per-episode autoregressive rollout MSE

**Step 3 — Visualize rollouts**
```bash
python models/scripts/visualize_swing.py \
    --data bat_data_100hz.pkl \
    --model logs/mocap_bc/<run>/policy_best.pt \
    --logdir logs/mocap_bc/<run>/ \
    --n_episodes 5 \
    --save_dir figures/
```
Generates per-episode 3-D trajectory plots and per-axis error curves.

---

## Simulation-based RL

### PyBullet environment

`PandaSwingBallEnv` simulates a 7-DOF Franka Panda arm with a bat in PyBullet.
Observation: joint angles (7) + bat tip position (3) + ball position (3) + hit flag (1) = **14-dim**.
Action: joint angle deltas (7-dim).

### PPO / SAC
```bash
python models/scripts/run_ppo.py
python models/scripts/run_sac.py
```

### Env-based BC (expert data required)
```bash
python models/scripts/run_bc.py \
    --expert_data <path_to_expert.pkl> \
    --exp_name bc_run \
    --env_name custom/PandaSwingBall-v0
```
