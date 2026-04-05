"""
sort_data.py

Merges joint angle and joint velocity CSVs for eligible right-handed swings,
computes per-joint angular accelerations, and saves the result to sorted_data.pkl.
"""

import csv
import pickle

velocities = "lead_elbow_angular_velocity_x,lead_elbow_angular_velocity_y,lead_elbow_angular_velocity_z,lead_hand_global_angular_velocity_x,lead_hand_global_angular_velocity_y,lead_hand_global_angular_velocity_z,lead_hip_angular_velocity_x,lead_hip_angular_velocity_y,lead_hip_angular_velocity_z,lead_knee_angular_velocity_x,lead_knee_angular_velocity_y,lead_knee_angular_velocity_z,lead_shoulder_angular_velocity_x,lead_shoulder_angular_velocity_y,lead_shoulder_angular_velocity_z,lead_shoulder_global_angular_velocity_x,lead_shoulder_global_angular_velocity_y,lead_shoulder_global_angular_velocity_z,lead_wrist_angular_velocity_x,lead_wrist_angular_velocity_y,lead_wrist_angular_velocity_z,pelvis_angular_velocity_x,pelvis_angular_velocity_y,pelvis_angular_velocity_z,rear_elbow_angular_velocity_x,rear_elbow_angular_velocity_y,rear_elbow_angular_velocity_z,rear_hand_global_angular_velocity_x,rear_hand_global_angular_velocity_y,rear_hand_global_angular_velocity_z,rear_hip_angular_velocity_x,rear_hip_angular_velocity_y,rear_hip_angular_velocity_z,rear_knee_angular_velocity_x,rear_knee_angular_velocity_y,rear_knee_angular_velocity_z,rear_shoulder_angular_velocity_x,rear_shoulder_angular_velocity_y,rear_shoulder_angular_velocity_z,rear_shoulder_global_angular_velocity_x,rear_shoulder_global_angular_velocity_y,rear_shoulder_global_angular_velocity_z,rear_wrist_angular_velocity_x,rear_wrist_angular_velocity_y,rear_wrist_angular_velocity_z,torso_angular_velocity_x,torso_angular_velocity_y,torso_angular_velocity_z,torso_pelvis_angular_velocity_x,torso_pelvis_angular_velocity_y,torso_pelvis_angular_velocity_z"
velocities = velocities.split(",")


def main():
    """Merge joint angles and velocities for eligible swings, compute accelerations, and save sorted_data.pkl."""
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

    final_data = {}
    count = 0
    keys_angles = []
    with open("../data/data/full_sig/joint_angles.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                keys_angles = line
                first = False
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                if not sess in final_data:
                    final_data[sess] = [line]
                else:
                    for i in range(len(final_data[sess])):
                        if float(final_data[sess][i][keys["time"]]) > float(line[keys["time"]]):
                            final_data[sess].insert(i, line)
                            break
                        if i == len(final_data[sess]) - 1:
                            final_data[sess].append(line)
    keys_velos = []
    with open("../data/data/full_sig/joint_velos.csv") as file:
        csv_reader = csv.reader(file, delimiter=',')
        first = True
        keys = {}
        for line in csv_reader:
            if count % 10000 == 0:
                print(count)
            count += 1
            if first:
                first = False
                keys_velos = line
                for i in range(len(line)):
                    keys[line[i]] = i
                continue
            sess = line[keys["session_swing"]]
            if sess in session_swings:
                for i in range(len(final_data[sess])):
                    if final_data[sess][i][keys["time"]] == line[keys["time"]]:
                        final_data[sess][i] += line
                        break
                    if i + 1 == len(final_data[sess]):
                        print(f"WARNING: no matching time entry for session {sess}")
    final_data["keys"] = keys_angles + keys_velos
    accel_names = [velo.replace("velocity", "accel") for velo in velocities]
    for i in range(len(final_data["keys"])):
        keys[final_data["keys"][i]] = i
    for sess in session_swings:
        print(sess)
        for i in range(len(final_data[sess])):
            for velo in velocities:
                start = float(final_data[sess][i][keys[velo]]) if final_data[sess][i][keys[velo]] != "" else 0.0
                end = 0.0
                if i + 1 < len(final_data[sess]) and final_data[sess][i + 1][keys[velo]] != "":
                    end = float(final_data[sess][i + 1][keys[velo]])
                delta_t = 1.0
                if i + 1 < len(final_data[sess]):
                    delta_t = float(final_data[sess][i+1][keys["time"]]) - float(final_data[sess][i][keys["time"]])
                accel = (end - start) / delta_t
                final_data[sess][i].append(accel)

    final_data["keys"] += accel_names

    with open('sorted_data.pkl', 'wb') as file:
        pickle.dump(final_data, file)

if __name__ == "__main__":
    main()