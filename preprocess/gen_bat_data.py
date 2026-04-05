"""
gen_bat_data.py

Reads eligible right-handed swings from eligible_swings.csv and the full
landmark CSV, then produces two pickle files:
  - bat_data.pkl        : raw per-frame bat data keyed by session_swing
  - bat_data_100hz.pkl  : downsampled to 100 Hz with observations, actions,
                          rewards, and terminals ready for BC training
"""

import numpy as np
import math
import pickle
import csv

MAX_TIMESTEPS = 200

BAT_SPEED_COEF = .1
HIT_COEF = 5
EUCLID_DIST_COEF = -2.5

def main():
    """Parse eligible swings and landmark data, then save bat_data.pkl and bat_data_100hz.pkl."""
    session_swings = set()
    with open("../eligible_swings.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            if first:
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            session_swings.add(line[keys["session_swing"]])
    count = 0
    final_data = {}
    with open("../data/data/full_sig/landmarks.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            #print(line)
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                if not sess in final_data.keys():
                    final_data[sess] = []
                time = float(line[keys["time"]])
                contact_time = float(line[keys["contact_time"]])
                sweet_spot_x = safe_float(final_data, sess, line[keys["sweet_spot_x"]], "x")
                sweet_spot_y = safe_float(final_data, sess, line[keys["sweet_spot_y"]], "y")
                sweet_spot_z = safe_float(final_data, sess, line[keys["sweet_spot_z"]], "z")
                
                lhjc_x = safe_float(final_data, sess,line[keys["lhjc_x"]], "lhjc_x") # left hand joint center
                lhjc_y = safe_float(final_data, sess,line[keys["lhjc_y"]], "lhjc_y")
                lhjc_z = safe_float(final_data, sess,line[keys["lhjc_z"]], "lhjc_z")

                lajc_x = safe_float(final_data, sess,line[keys["lajc_x"]], "lajc_x")
                lajc_y = safe_float(final_data, sess,line[keys["lajc_y"]], "lajc_y")
                lajc_z = safe_float(final_data, sess,line[keys["lajc_z"]], "lajc_z")
                rajc_x = safe_float(final_data, sess,line[keys["rajc_x"]], "rajc_x")
                rajc_y = safe_float(final_data, sess,line[keys["rajc_y"]], "rajc_y")
                rajc_z = safe_float(final_data, sess,line[keys["rajc_z"]], "rajc_z")

                x_ang, y_ang, z_ang = calc_pose(np.array([lhjc_x, lhjc_y, lhjc_z]), np.array([sweet_spot_x, sweet_spot_y, sweet_spot_z]))
                x_vel = 0
                y_vel = 0
                z_vel = 0
                x_ang_vel = 0
                y_ang_vel = 0
                z_ang_vel = 0

                if(len(final_data[sess]) != 0):
                    delta_t = time - final_data[sess][-1]["time"]
                    x_vel = (sweet_spot_x - final_data[sess][-1]["x"]) / delta_t
                    y_vel = (sweet_spot_y - final_data[sess][-1]["y"]) / delta_t
                    z_vel = (sweet_spot_z - final_data[sess][-1]["z"]) / delta_t

                    x_ang_vel = (x_ang - final_data[sess][-1]["x_ang"]) / delta_t
                    y_ang_vel = (y_ang - final_data[sess][-1]["y_ang"]) / delta_t
                    z_ang_vel = (z_ang - final_data[sess][-1]["z_ang"]) / delta_t
                next_entry = {
                    "time": time,
                    "contact_time": contact_time,
                    "x": sweet_spot_x,
                    "y": sweet_spot_y,
                    "z": sweet_spot_z,
                    "x_ang": x_ang,
                    "y_ang": y_ang,
                    "z_ang": z_ang,
                    "x_vel": x_vel,
                    "y_vel": y_vel,
                    "z_vel": z_vel,
                    "x_ang_vel": x_ang_vel,
                    "y_ang_vel": y_ang_vel,
                    "z_ang_vel": z_ang_vel,
                    "lhjc_x": lhjc_x,
                    "lhjc_y": lhjc_y,
                    "lhjc_z": lhjc_z,
                    "lajc_x": lajc_x,
                    "lajc_y": lajc_y,
                    "lajc_z": lajc_z,
                    "rajc_x": rajc_x,
                    "rajc_y": rajc_y,
                    "rajc_z": rajc_z
                }
                final_data[sess].append(next_entry)
    with open('../bat_data.pkl', 'wb') as file:
        pickle.dump(final_data, file)

    low_hz = []
    for sess in final_data.keys():
        print(sess)
        contact_time = final_data[sess][0]["contact_time"]
        interval = 0.01
        index = 0
        episodes =[]
        ball_pos = None
        while index < len(final_data[sess]):
            collect = []
            while  index < len(final_data[sess]) and final_data[sess][index]["time"] < interval:
                collect.append(final_data[sess][index])
                #print("TRUE TIME:", collect[-1]["time"])
                index += 1           
            averaged = average(collect)
            if contact_time < averaged["time"]:
                #print("HIT FOUND!")
                contact_time = 999999
                ball_pos = [averaged["x"], averaged["y"], averaged["z"]]
            #else:
                #print(contact_time, averaged["time"])
            episodes.append(averaged)
            interval += 0.01
        ep_index = 0
        obs = []
        next_obs = []
        actions = []
        rewards = []
        terminals = []
        while ep_index < MAX_TIMESTEPS and ep_index < len(episodes):
            obs.append(gen_observation(episodes, ep_index, ball_pos))
            next_obs.append(gen_observation(episodes, ep_index + 1, ball_pos))
            actions.append(next_obs[-1][0][:6])
            reward = 0
            if next_obs[-1][0][-1] != obs[-1][0][-1]: # hit change of state
                reward += HIT_COEF * 1

            bat_velo = (next_obs[-1][0][6]**2 + next_obs[-1][0][7]**2 + next_obs[-1][0][8]**2)**.5
            reward += bat_velo * BAT_SPEED_COEF
            if obs[-1][0][-1] != 1:
                dist = ((next_obs[-1][0][0] - next_obs[-1][0][-4])**2 + (next_obs[-1][0][1] - next_obs[-1][0][-3]) + (next_obs[-1][0][2] - next_obs[-1][0][-2]))**.5
                reward += EUCLID_DIST_COEF * dist
            rewards.append(reward)
            if ep_index == MAX_TIMESTEPS -1 or ep_index == len(episodes) -1:
                terminals.append(1)
            else:
                terminals.append(0)
            ep_index += 1
        low_hz.append({
            "observations": obs,
            "next_observations": next_obs,
            "rewards": rewards,
            "actions": actions,
            "terminals": terminals
        })
    with open('../bat_data_100hz.pkl', 'wb') as file:
        pickle.dump(low_hz, file)

def gen_observation(episodes, index, ball_pos, k=5):
    """
    Build a stacked observation from k consecutive frames ending at index.

    Each frame contains sweet-spot position/orientation/velocity (12 values),
    ball position (3 values), and a binary hit flag (1 value). Frames before
    the start of the episode are filled by repeating the earliest available frame.
    """
    ret = []
    
    for i in range(k):
        if(index - i < 0):
            ret.append(ret[-1])
            continue
        idx = index
        if idx >= len(episodes):
            idx = len(episodes) - 1
        time_step = episodes[idx - i]
        hit = 0
        if time_step["contact_time"] < time_step["time"]:
            hit = 1
        vals = [
            time_step["x"],
            time_step["y"],
            time_step["z"],
            time_step["x_ang"],
            time_step["y_ang"],
            time_step["z_ang"],
            time_step["x_vel"],
            time_step["y_vel"],
            time_step["z_vel"],
            time_step["x_ang_vel"],
            time_step["y_ang_vel"],
            time_step["z_ang_vel"],
            ball_pos[0],
            ball_pos[1],
            ball_pos[2],
            hit
        ]
        ret.append(vals)
    return ret



def average(collect):
    """Return a dict whose values are the mean of each key across all frames."""
    out = {}
    for key in collect[0].keys():
        total = sum(frame[key] for frame in collect)
        out[key] = total / len(collect)
    return out

# In case there are gaps in the data, fall back to the previous valid value
def safe_float(final_data, sess, value_str, key):
    """Parse value_str as float; on failure return the last stored value for key."""
    try:
        return float(value_str)
    except ValueError:
        return final_data[sess][-1][key]

def calc_pose(left_hand, bat_sweet_spot):
    """
    Compute Euler-style orientation angles (x, y, z) for the bat axis defined
    by the vector from left_hand to bat_sweet_spot.
    """
    axis = bat_sweet_spot - left_hand
    axis = axis / (np.sqrt(np.dot(axis, axis)))
    x = np.array([1,0,0])
    y = np.array([0,1,0])
    z = np.array([0,0,1])
    proj_yz = proj(axis, y, z)
    proj_yz = np.array([proj_yz[1], proj_yz[2]])
    proj_yx = proj(axis, x, y)
    proj_yx = np.array([proj_yx[0], proj_yx[1]])
    y_2d_proj_yz = np.array([1,0])
    y_2d_proj_yx = np.array([0,1])
    multiplier = 1
    if cross_prod(proj_yz, y_2d_proj_yz) > 0:
        multiplier = -1
    angle_x = multiplier * cos_deg(y_2d_proj_yz, proj_yz)
    if cross_prod(proj_yx, y_2d_proj_yx) > 0:
        multiplier = -1
    else:
        multiplier = 1
    angle_z = multiplier * cos_deg(y_2d_proj_yx, proj_yx)
    angle_y = 0
    return angle_x, angle_y, angle_z

def proj(axis, axis_1, axis_2):
    """Project axis onto the plane spanned by axis_1 and axis_2."""
    a1 = (np.dot(axis, axis_1) / np.dot(axis_1, axis_1)) * axis_1
    a2 = (np.dot(axis, axis_2) / np.dot(axis_2, axis_2)) * axis_2
    return a1 + a2

def cos_deg(a, b):
    """Return the angle in degrees between 2-D vectors a and b."""
    return (180 / math.pi) * np.arccos(np.dot(a, b) / (np.sqrt(np.dot(a,a)) * np.sqrt(np.dot(b,b))))

def cross_prod(a, b):
    """Return the scalar 2-D cross product of vectors a and b."""
    return (a[0] * b[1]) - (a[1] * b[0])

if __name__ == "__main__":
    main()