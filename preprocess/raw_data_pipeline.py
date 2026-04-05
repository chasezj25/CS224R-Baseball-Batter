"""
raw_data_pipeline.py

Reads motion-capture baseball swing data directly from the raw zip archives in
`data/data/full_sig/`, without requiring any intermediate pickle files.

The module exposes a single class, RawMocapPipeline, whose `iter_episodes()`
method is a generator that yields processed swing-episode dicts in the same
format as the bat_data_100hz.pkl episodes produced by preprocess/gen_bat_data.py.

Episode dict keys
-----------------
observations     : list of T elements, each a list of k=5 stacked frames.
                   Each frame is a 16-element list:
                     [x, y, z, x_ang, y_ang, z_ang,
                      x_vel, y_vel, z_vel,
                      x_ang_vel, y_ang_vel, z_ang_vel,
                      ball_pos_x, ball_pos_y, ball_pos_z,
                      hit_flag]
next_observations: same, shifted by one timestep
actions          : list of T elements, each [x, y, z, x_ang, y_ang, z_ang]
                   (the next-step bat sweet-spot pose)
rewards          : list of T floats
terminals        : list of T ints (0 or 1)

Usage
-----
    from preprocess.raw_data_pipeline import RawMocapPipeline

    pipeline = RawMocapPipeline(
        data_dir="data/data/full_sig",
        eligible_swings_path="eligible_swings.csv",   # optional
    )
    for episode in pipeline.iter_episodes():
        obs = episode["observations"]
        ...
"""

import csv
import io
import math
import os
import zipfile
from collections import defaultdict

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants matching gen_bat_data.py
# ──────────────────────────────────────────────────────────────────────────────

_CONTACT_FOUND_SENTINEL = 999_999.0  # Replaced contact_time after hit is detected
MAX_TIMESTEPS    = 200
HISTORY_LEN      = 5       # k stacked frames per observation
TARGET_HZ        = 100     # 100 Hz downsampling target
INTERVAL         = 1.0 / TARGET_HZ

BAT_SPEED_COEF   = 0.1
HIT_COEF         = 5.0
EUCLID_DIST_COEF = -2.5


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers (copied from gen_bat_data.py for consistency)
# ──────────────────────────────────────────────────────────────────────────────

def _proj(axis, axis_1, axis_2):
    a1 = (np.dot(axis, axis_1) / np.dot(axis_1, axis_1)) * axis_1
    a2 = (np.dot(axis, axis_2) / np.dot(axis_2, axis_2)) * axis_2
    return a1 + a2


def _cos_deg(a, b):
    denom = math.sqrt(np.dot(a, a)) * math.sqrt(np.dot(b, b))
    if denom < 1e-10:
        return 0.0
    return (180.0 / math.pi) * math.acos(
        max(-1.0, min(1.0, np.dot(a, b) / denom))
    )


def _cross_prod(a, b):
    return a[0] * b[1] - a[1] * b[0]


def calc_pose(left_hand, bat_sweet_spot):
    axis = bat_sweet_spot - left_hand
    norm = math.sqrt(np.dot(axis, axis))
    if norm < 1e-10:
        return 0.0, 0.0, 0.0
    axis = axis / norm

    x = np.array([1, 0, 0], dtype=float)
    y = np.array([0, 1, 0], dtype=float)
    z = np.array([0, 0, 1], dtype=float)

    proj_yz = _proj(axis, y, z)
    proj_yz = np.array([proj_yz[1], proj_yz[2]])
    proj_yx = _proj(axis, x, y)
    proj_yx = np.array([proj_yx[0], proj_yx[1]])

    y_2d_proj_yz = np.array([1, 0], dtype=float)
    y_2d_proj_yx = np.array([0, 1], dtype=float)

    multiplier = -1 if _cross_prod(proj_yz, y_2d_proj_yz) > 0 else 1
    angle_x = multiplier * _cos_deg(y_2d_proj_yz, proj_yz)
    multiplier = -1 if _cross_prod(proj_yx, y_2d_proj_yx) > 0 else 1
    angle_z = multiplier * _cos_deg(y_2d_proj_yx, proj_yx)
    angle_y = 0.0
    return angle_x, angle_y, angle_z


# ──────────────────────────────────────────────────────────────────────────────
# CSV row field helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_float(val_str, fallback):
    try:
        return float(val_str)
    except (ValueError, TypeError):
        return fallback


# ──────────────────────────────────────────────────────────────────────────────
# Observation builder (mirrors gen_bat_data.py / generate_demo_data.py)
# ──────────────────────────────────────────────────────────────────────────────

def _gen_observation(episodes, index, ball_pos, k=HISTORY_LEN):
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


def _average_frames(frames):
    """Average a list of frame dicts (for downsampling to TARGET_HZ)."""
    if len(frames) == 0:
        return None
    if len(frames) == 1:
        return dict(frames[0])
    out = {}
    for key in frames[0]:
        out[key] = sum(f[key] for f in frames) / len(frames)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline class
# ──────────────────────────────────────────────────────────────────────────────

class RawMocapPipeline:
    """
    Streams baseball swing episodes directly from the raw zip archives.

    Parameters
    ----------
    data_dir : str
        Directory containing the full_sig zip files (landmarks.zip, …).
        Defaults to "data/data/full_sig" relative to the current working
        directory if not provided.
    eligible_swings_path : str | None
        Path to eligible_swings.csv (produced by preprocess/filter_data.py).
        If provided, only swings listed there are processed.
        If None and metadata_path is also None, all swings are processed.
    metadata_path : str | None
        Path to metadata.csv.  If eligible_swings_path is not provided but
        metadata_path is, the pipeline filters out left-handed swings
        (hitter_side == "L") on-the-fly.
    max_timesteps : int
        Maximum number of 100 Hz timesteps per episode.
    """

    def __init__(
        self,
        data_dir=None,
        eligible_swings_path=None,
        metadata_path=None,
        max_timesteps=MAX_TIMESTEPS,
    ):
        if data_dir is None:
            data_dir = os.path.join("data", "data", "full_sig")
        self.data_dir = data_dir
        self.landmarks_zip = os.path.join(data_dir, "landmarks.zip")
        self.eligible_swings_path = eligible_swings_path
        self.metadata_path = metadata_path
        self.max_timesteps = max_timesteps

        if not os.path.isfile(self.landmarks_zip):
            raise FileNotFoundError(
                f"landmarks.zip not found at {self.landmarks_zip}. "
                "Make sure data_dir points to the full_sig directory."
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def iter_episodes(self):
        """
        Yield processed swing-episode dicts, one per eligible swing.

        Each dict has the same structure as a bat_data_100hz.pkl episode:
        {
            'observations': list[list[list[float]]],  # T × k × 16
            'next_observations': list[list[list[float]]],
            'actions': list[list[float]],             # T × 6
            'rewards': list[float],                   # T
            'terminals': list[int],                   # T
        }
        """
        eligible = self._load_eligible_swings()
        raw_per_swing = self._stream_landmarks(eligible)
        for sess, raw_frames in raw_per_swing.items():
            episode = self._build_episode(raw_frames)
            if episode is not None:
                yield episode

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_eligible_swings(self):
        """Return a set of eligible session_swing IDs, or None (= all)."""
        if self.eligible_swings_path and os.path.isfile(self.eligible_swings_path):
            swings = set()
            with open(self.eligible_swings_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    swings.add(row["session_swing"])
            return swings

        if self.metadata_path and os.path.isfile(self.metadata_path):
            swings = set()
            with open(self.metadata_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("hitter_side", "R") != "L":
                        swings.add(row["session_swing"])
            return swings

        return None  # process all swings

    def _stream_landmarks(self, eligible):
        """
        Stream landmarks.csv from inside landmarks.zip and collect per-swing
        raw-frame dicts.  Returns an OrderedDict {session_swing: [frame, ...]}.
        """
        per_swing = defaultdict(list)
        fallback = {}   # last known value per (sess, key) for gap-filling

        with zipfile.ZipFile(self.landmarks_zip, "r") as zf:
            with zf.open("landmarks.csv") as raw_bytes:
                text = io.TextIOWrapper(raw_bytes, encoding="utf-8")
                reader = csv.DictReader(text)

                for row in reader:
                    sess = row["session_swing"]
                    if eligible is not None and sess not in eligible:
                        continue

                    # Parse required fields with fallback for gaps
                    fb = fallback.setdefault(sess, {})

                    def get(key):
                        val = _try_float(row.get(key, ""), fb.get(key, 0.0))
                        fb[key] = val
                        return val

                    t            = get("time")
                    contact_time = get("contact_time")
                    ss_x         = get("sweet_spot_x")
                    ss_y         = get("sweet_spot_y")
                    ss_z         = get("sweet_spot_z")
                    lhjc_x       = get("lhjc_x")
                    lhjc_y       = get("lhjc_y")
                    lhjc_z       = get("lhjc_z")
                    lajc_x       = get("lajc_x")
                    lajc_y       = get("lajc_y")
                    lajc_z       = get("lajc_z")
                    rajc_x       = get("rajc_x")
                    rajc_y       = get("rajc_y")
                    rajc_z       = get("rajc_z")

                    x_ang, y_ang, z_ang = calc_pose(
                        np.array([lhjc_x, lhjc_y, lhjc_z]),
                        np.array([ss_x, ss_y, ss_z]),
                    )

                    frames = per_swing[sess]
                    if frames:
                        prev = frames[-1]
                        dt   = max(t - prev["time"], 1e-6)
                        x_vel      = (ss_x  - prev["x"])     / dt
                        y_vel      = (ss_y  - prev["y"])     / dt
                        z_vel      = (ss_z  - prev["z"])     / dt
                        x_ang_vel  = (x_ang - prev["x_ang"]) / dt
                        y_ang_vel  = (y_ang - prev["y_ang"]) / dt
                        z_ang_vel  = (z_ang - prev["z_ang"]) / dt
                    else:
                        x_vel = y_vel = z_vel = 0.0
                        x_ang_vel = y_ang_vel = z_ang_vel = 0.0

                    frames.append({
                        "time": t, "contact_time": contact_time,
                        "x": ss_x, "y": ss_y, "z": ss_z,
                        "x_ang": x_ang, "y_ang": y_ang, "z_ang": z_ang,
                        "x_vel": x_vel, "y_vel": y_vel, "z_vel": z_vel,
                        "x_ang_vel": x_ang_vel,
                        "y_ang_vel": y_ang_vel,
                        "z_ang_vel": z_ang_vel,
                        "lhjc_x": lhjc_x, "lhjc_y": lhjc_y, "lhjc_z": lhjc_z,
                        "lajc_x": lajc_x, "lajc_y": lajc_y, "lajc_z": lajc_z,
                        "rajc_x": rajc_x, "rajc_y": rajc_y, "rajc_z": rajc_z,
                    })

        return per_swing

    def _build_episode(self, raw_frames):
        """Downsample to TARGET_HZ and build (obs, action, reward, terminal) lists."""
        if not raw_frames:
            return None

        contact_time = raw_frames[0]["contact_time"]
        interval     = INTERVAL
        index        = 0
        episodes     = []
        ball_pos     = None

        # Downsample: average all raw frames within each 10ms bin
        while index < len(raw_frames):
            collect = []
            while index < len(raw_frames) and raw_frames[index]["time"] < interval:
                collect.append(raw_frames[index])
                index += 1
            if not collect:
                # Skip empty bins (shouldn't normally happen)
                interval += INTERVAL
                continue
            averaged = _average_frames(collect)
            if contact_time < averaged["time"] and ball_pos is None:
                ball_pos = [averaged["x"], averaged["y"], averaged["z"]]
                contact_time = _CONTACT_FOUND_SENTINEL  # mark as found
            episodes.append(averaged)
            interval += INTERVAL

        if ball_pos is None:
            # No contact detected: use last known position
            ball_pos = [episodes[-1]["x"], episodes[-1]["y"], episodes[-1]["z"]]

        obs_list, next_obs_list, actions, rewards, terminals = [], [], [], [], []
        ep_steps = min(self.max_timesteps, len(episodes))

        for ep_index in range(ep_steps):
            obs_t  = _gen_observation(episodes, ep_index,     ball_pos)
            obs_t1 = _gen_observation(episodes, ep_index + 1, ball_pos)

            obs_list.append(obs_t)
            next_obs_list.append(obs_t1)
            actions.append(obs_t1[0][:6])

            # Reward (same formula as gen_bat_data.py)
            reward = 0.0
            if obs_t1[0][-1] != obs_t[0][-1]:
                reward += HIT_COEF
            bat_velo = math.sqrt(
                obs_t1[0][6] ** 2 + obs_t1[0][7] ** 2 + obs_t1[0][8] ** 2
            )
            reward += bat_velo * BAT_SPEED_COEF
            if obs_t[0][-1] != 1:
                dist = math.sqrt(
                    (obs_t1[0][0] - obs_t1[0][-4]) ** 2
                    + (obs_t1[0][1] - obs_t1[0][-3]) ** 2
                    + (obs_t1[0][2] - obs_t1[0][-2]) ** 2
                )
                reward += EUCLID_DIST_COEF * dist
            rewards.append(reward)

            terminal = 1 if ep_index == ep_steps - 1 else 0
            terminals.append(terminal)

        return {
            "observations":      obs_list,
            "next_observations": next_obs_list,
            "actions":           actions,
            "rewards":           rewards,
            "terminals":         terminals,
        }
