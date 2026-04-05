"""
gen_sim_expert_data.py

Reads raw motion-capture baseball swing data directly from the landmarks.zip
archive (no intermediate pickle files), maps each bat trajectory to Panda arm
joint angles using PyBullet inverse kinematics, and writes an expert
demonstration file (expert_sim_data.pkl) in the exact format expected by
models/infrastructure/bc_trainer.py and the PandaSwingBallEnv.

Simulation observation space (14-dim, then clipped to [-1, 1] by /10)
----------------------------------------------------------------------
  [0:7]   joint positions (joints 0-6 of panda_arm_bat.urdf)
  [7:10]  bat tip position (link 7 world coordinates)
  [10:13] ball position (world coordinates)
  [13]    ball_hit flag (0.0 or 1.0)

Action space (7-dim)
--------------------
  Joint angle deltas (added to current joint positions at each step)

Usage
-----
    python preprocess/gen_sim_expert_data.py \\
        --data_dir  data/data/full_sig \\
        --eligible  eligible_swings.csv \\
        --output    expert_sim_data.pkl \\
        --max_swings 50

    # Then train BC on the generated expert data:
    python models/scripts/run_bc.py \\
        --expert_data expert_sim_data.pkl \\
        --exp_name sim_bc_from_raw \\
        --env_name custom/PandaSwingBall-v0
"""

import argparse
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("."))

# ──────────────────────────────────────────────────────────────────────────────
# Simulation constants (must match PandaSwingBallEnv)
# ──────────────────────────────────────────────────────────────────────────────

OB_DIM              = 14     # joint_angles(7) + bat_pos(3) + ball_pos(3) + hit(1)
AC_DIM              = 7      # joint angle deltas
NUM_JOINTS          = 7      # joints 0-6 in the URDF
EE_LINK_INDEX       = 7      # bat-tip link index
SIM_BALL_POS        = np.array([0.0, -1.0, 0.7], dtype=np.float64)

# Closest point in link-7 workspace to ball (empirically determined)
SIM_CONTACT_ZONE    = np.array([0.0, -0.87, 0.67], dtype=np.float64)

# Maximum radius of scaled MoCap trajectory around contact zone
SIM_TRAJ_RADIUS     = 0.25   # metres

# Reward coefficients (must match PandaSwingBallEnv._compute_reward)
HIT_REWARD          = 100.0
DISTANCE_WEIGHT     = -2.0
BAT_SPEED_WEIGHT    = 0.5
LEFT_ROTATION_WEIGHT = 1.0

OBS_NORM_SCALE      = 10.0   # env normalises by dividing by this

# IK convergence
IK_MAX_ITER         = 200
IK_RESIDUAL_THRESH  = 1e-4


# ──────────────────────────────────────────────────────────────────────────────
# PyBullet IK helper
# ──────────────────────────────────────────────────────────────────────────────

class PandaIKSolver:
    """
    Thin wrapper around PyBullet that provides inverse kinematics for the
    panda_arm_bat URDF.  Runs in DIRECT (headless) mode.
    """

    def __init__(self, urdf_path):
        import pybullet as p
        import pybullet_data
        self._p = p
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF(urdf_path, useFixedBase=True)
        self.num_joints = NUM_JOINTS
        self.ee_link    = EE_LINK_INDEX

        # Collect joint limits for the controllable joints (1-7)
        self._ll = []
        self._ul = []
        self._jr = []
        self._rp = []
        for i in range(1, self.num_joints + 1):
            info = p.getJointInfo(self.robot, i)
            lo, hi = info[8], info[9]
            if lo >= hi:          # fixed joint — treat as free
                lo, hi = -math.pi, math.pi
            self._ll.append(lo)
            self._ul.append(hi)
            self._jr.append(hi - lo)
            self._rp.append((lo + hi) / 2.0)

        # Warm-start with a neutral pose
        self._current_angles = list(self._rp)

    def solve(self, target_pos):
        """
        Compute joint angles (joints 1-7) that place link EE_LINK_INDEX at
        target_pos.  Updates the internal current-angle state so successive
        calls benefit from warm-starting.

        Returns
        -------
        angles : np.ndarray, shape (NUM_JOINTS,)
            Joint angles for joints 1-7 (indices match the env's range(7)).
        achieved_pos : np.ndarray, shape (3,)
            Actual end-effector position after applying the IK solution.
        """
        p = self._p

        # Set robot to current warm-start
        for i, ang in enumerate(self._current_angles):
            p.resetJointState(self.robot, i + 1, ang)

        raw = p.calculateInverseKinematics(
            self.robot,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=target_pos.tolist(),
            lowerLimits=self._ll,
            upperLimits=self._ul,
            jointRanges=self._jr,
            restPoses=self._rp,
            maxNumIterations=IK_MAX_ITER,
            residualThreshold=IK_RESIDUAL_THRESH,
        )
        angles = np.array(raw[:self.num_joints], dtype=np.float64)

        # Apply result and measure achieved position
        for i, ang in enumerate(angles):
            p.resetJointState(self.robot, i + 1, ang)
        p.stepSimulation()
        achieved = np.array(p.getLinkState(self.robot, self.ee_link)[0])

        # Update warm-start
        self._current_angles = angles.tolist()
        return angles, achieved

    def get_bat_pos(self, angles):
        """Return link-7 world position for the given joint angles."""
        p = self._p
        for i, ang in enumerate(angles):
            p.resetJointState(self.robot, i + 1, ang)
        p.stepSimulation()
        return np.array(p.getLinkState(self.robot, self.ee_link)[0])

    def reset_warm_start(self):
        """Reset warm-start to neutral pose."""
        self._current_angles = list(self._rp)

    def close(self):
        self._p.disconnect(self.client)


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transform: MoCap → Simulation space
# ──────────────────────────────────────────────────────────────────────────────

def transform_mocap_trajectory(bat_positions, contact_position):
    """
    Transform a MoCap bat-sweet-spot trajectory so it is centred on the
    simulation's contact zone and scaled to fit within SIM_TRAJ_RADIUS.

    The transformation is:
        1. Translate so contact_position → origin.
        2. Scale so max distance from origin ≤ SIM_TRAJ_RADIUS.
        3. Translate so origin → SIM_CONTACT_ZONE.

    Parameters
    ----------
    bat_positions   : np.ndarray (T, 3)
    contact_position: np.ndarray (3,)  — MoCap position at bat-ball contact

    Returns
    -------
    sim_positions : np.ndarray (T, 3)  — in simulation world coordinates
    scale         : float              — applied scale factor
    """
    centred = bat_positions - contact_position
    max_dist = np.max(np.linalg.norm(centred, axis=1))
    scale = 1.0
    if max_dist > SIM_TRAJ_RADIUS:
        scale = SIM_TRAJ_RADIUS / max_dist
        centred = centred * scale
    sim_positions = centred + SIM_CONTACT_ZONE
    return sim_positions, scale


# ──────────────────────────────────────────────────────────────────────────────
# Observation / reward builders (matching PandaSwingBallEnv exactly)
# ──────────────────────────────────────────────────────────────────────────────

def build_observation(joint_angles, bat_pos, ball_pos, ball_hit):
    """Build and normalise a 14-dim observation vector (as in the sim env)."""
    obs = np.concatenate([
        joint_angles,        # (7,)
        bat_pos,             # (3,)
        ball_pos,            # (3,)
        [float(ball_hit)],   # (1,)
    ]).astype(np.float32)
    obs = np.clip(obs / OBS_NORM_SCALE, -1.0, 1.0)
    return obs


def compute_sim_reward(bat_pos, prev_bat_pos, ball_pos, ball_hit, dt=0.01):
    """
    Compute reward following PandaSwingBallEnv._compute_reward logic.

    Note: the ball_velocity term is omitted (not available in a pure IK
    pipeline without a running physics simulation), and the angular velocity
    rotation-encouragement reward is also omitted for the same reason.
    """
    distance   = np.linalg.norm(bat_pos - ball_pos)
    bat_speed  = (np.linalg.norm(bat_pos - prev_bat_pos) / dt
                  if prev_bat_pos is not None else 0.0)

    if not ball_hit:
        if distance < 0.1:
            return (HIT_REWARD + BAT_SPEED_WEIGHT * bat_speed), True
        return DISTANCE_WEIGHT * distance + BAT_SPEED_WEIGHT * bat_speed, False
    else:
        ball_x_dist = abs(ball_pos[0])
        return ball_x_dist, ball_hit


# ──────────────────────────────────────────────────────────────────────────────
# Core conversion: one MoCap episode → simulation expert path
# ──────────────────────────────────────────────────────────────────────────────

def mocap_episode_to_sim_path(episode, ik_solver):
    """
    Convert a MoCap episode (from RawMocapPipeline) to a simulation expert
    trajectory suitable for bc_trainer.py.

    Parameters
    ----------
    episode   : dict — output of RawMocapPipeline.iter_episodes()
    ik_solver : PandaIKSolver

    Returns
    -------
    path : dict with keys 'observations', 'actions', 'rewards',
           'next_observations', 'terminals'   (all as np.ndarray)
    None if the episode is empty or IK fails catastrophically.
    """
    raw_obs   = episode["observations"]
    raw_acts  = episode["actions"]
    T         = len(raw_acts)
    if T == 0:
        return None

    # Extract bat sweet-spot positions (index 0-2 of each frame's most-recent slot)
    bat_positions_mocap = np.array(
        [[o[0][0], o[0][1], o[0][2]] for o in raw_obs], dtype=np.float64
    )

    # Contact position: the ball_pos stored in observations (indices 12-14)
    contact_mocap = np.array(raw_obs[0][0][12:15], dtype=np.float64)

    # Transform trajectory to simulation coordinates
    sim_positions, _ = transform_mocap_trajectory(bat_positions_mocap, contact_mocap)

    # Reset IK warm-start for this new swing
    ik_solver.reset_warm_start()

    # Compute joint angles via IK for each timestep
    joint_angles_seq = []
    bat_pos_seq      = []
    for pos in sim_positions:
        angles, achieved = ik_solver.solve(pos)
        joint_angles_seq.append(angles)
        bat_pos_seq.append(achieved)

    # Determine hit flag from MoCap
    hit_flags = [int(o[0][-1]) for o in raw_obs]

    observations      = []
    next_observations = []
    actions           = []
    rewards           = []
    terminals         = []

    ball_hit = False
    prev_bat_pos = None

    for t in range(T):
        ja  = joint_angles_seq[t]
        bp  = bat_pos_seq[t]
        ball_pos = SIM_BALL_POS

        if not ball_hit and hit_flags[t] == 1:
            ball_hit = True

        obs = build_observation(ja, bp, ball_pos, ball_hit)
        observations.append(obs)

        # Next step
        if t + 1 < T:
            ja_next = joint_angles_seq[t + 1]
            bp_next = bat_pos_seq[t + 1]
            hf_next = ball_hit or (hit_flags[t + 1] == 1)
        else:
            ja_next = ja
            bp_next = bp
            hf_next = ball_hit
        next_obs = build_observation(ja_next, bp_next, ball_pos, hf_next)
        next_observations.append(next_obs)

        # Action: delta in joint angles
        action = (ja_next - ja).astype(np.float32)
        actions.append(action)

        # Reward
        rew, ball_hit = compute_sim_reward(bp, prev_bat_pos, ball_pos, ball_hit)
        rewards.append(float(rew))
        prev_bat_pos = bp

        terminals.append(1 if t == T - 1 else 0)

    return {
        "observations":      np.array(observations,      dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "actions":           np.array(actions,           dtype=np.float32),
        "rewards":           np.array(rewards,           dtype=np.float32),
        "terminals":         np.array(terminals,         dtype=np.int32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation-compatible expert data from raw MoCap zips."
    )
    parser.add_argument(
        "--data_dir", "-d", type=str,
        default=os.path.join("data", "data", "full_sig"),
        help="Directory containing landmarks.zip (and other full_sig zips).",
    )
    parser.add_argument(
        "--eligible", "-e", type=str, default=None,
        help="Path to eligible_swings.csv (optional filter).",
    )
    parser.add_argument(
        "--metadata", "-m", type=str, default=None,
        help="Path to metadata.csv (used when --eligible is not provided).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="expert_sim_data.pkl",
        help="Output pkl file path.",
    )
    parser.add_argument(
        "--max_swings", type=int, default=None,
        help="Maximum number of swings to process (useful for testing).",
    )
    parser.add_argument(
        "--urdf", type=str,
        default=os.path.join("models", "envs", "panda_arm_bat.urdf"),
        help="Path to panda_arm_bat.urdf.",
    )
    args = parser.parse_args()

    # ── Imports here to keep them out of the module-level namespace ──────────
    from preprocess.raw_data_pipeline import RawMocapPipeline

    print(f"Loading raw data from: {args.data_dir}")
    pipeline = RawMocapPipeline(
        data_dir=args.data_dir,
        eligible_swings_path=args.eligible,
        metadata_path=args.metadata,
    )

    print(f"Initialising PyBullet IK solver with URDF: {args.urdf}")
    ik_solver = PandaIKSolver(urdf_path=args.urdf)

    expert_paths = []
    n_processed = 0
    n_skipped   = 0

    print("Processing swings …")
    for episode in pipeline.iter_episodes():
        if args.max_swings is not None and n_processed >= args.max_swings:
            break

        path = mocap_episode_to_sim_path(episode, ik_solver)
        if path is None:
            n_skipped += 1
            continue

        expert_paths.append(path)
        n_processed += 1
        if n_processed % 10 == 0:
            print(f"  Processed {n_processed} swings …")

    ik_solver.close()

    if not expert_paths:
        print("ERROR: No valid paths generated. Check data paths.")
        sys.exit(1)

    total_steps = sum(len(p["rewards"]) for p in expert_paths)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "wb") as f:
        pickle.dump(expert_paths, f)

    print(
        f"\nSaved {len(expert_paths)} expert trajectories "
        f"({total_steps} total timesteps) → {args.output}\n"
        f"Skipped {n_skipped} episodes (empty or IK failure).\n"
        f"\nTo train BC on this data:\n"
        f"  python models/scripts/run_bc.py \\\n"
        f"    --expert_data {args.output} \\\n"
        f"    --exp_name sim_bc_from_raw \\\n"
        f"    --env_name custom/PandaSwingBall-v0\n"
    )


if __name__ == "__main__":
    main()
