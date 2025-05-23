

import pandas as pd
import numpy as np
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R
import time

# Load motion capture swing dataset
df = pd.read_csv("../rh_swing_data.csv")
session_ids = df["session_swing"].unique()

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path("humanoid_mocap_model.xml")
data = mujoco.MjData(model)

# Mapping dataset joint labels to mocap body names in XML
JOINT_MAPPING = {
    "lead_shoulder": "mocap_shoulder_left",
    "lead_elbow": "mocap_elbow_left",
    "lead_hip": "mocap_hip_left",
    "lead_knee": "mocap_knee_left",
    "rear_shoulder": "mocap_shoulder_right",
    "rear_elbow": "mocap_elbow_right",
    "rear_hip": "mocap_hip_right",
    "rear_knee": "mocap_knee_right",
    "torso": "mocap_torso",
    "pelvis": "mocap_pelvis"
}

AXES = ["x", "y", "z"]

def set_mocap_rotation(body_name, euler_angles_deg):
    quat = R.from_euler('xyz', euler_angles_deg, degrees=True).as_quat()
    quat = np.roll(quat, 1)  # MuJoCo uses [w, x, y, z]
    mocap_id = model.body(body_name).mocapid
    if mocap_id >= 0:
        data.mocap_quat[mocap_id] = quat

def set_mocap_position(body_name, pos_xyz):
    mocap_id = model.body(body_name).mocapid
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = np.array(pos_xyz)

# Viewer loop
with viewer.launch_passive(model, data) as v:
    for session_id in session_ids:
        print(f"Animating session: {session_id}")
        session_df = df[df["session_swing"] == session_id].sort_values(by="time")

        for _, row in session_df.iterrows():
            for dataset_joint, mocap_body in JOINT_MAPPING.items():
                # Set orientation
                angles = [row.get(f"{dataset_joint}_angle_{axis}", 0) for axis in AXES]
                set_mocap_rotation(mocap_body, angles)

                # Set position (if available)
                pos_keys = [f"{dataset_joint}_x", f"{dataset_joint}_y", f"{dataset_joint}_z"]
                if all(key in row for key in pos_keys):
                    pos = [row[k] for k in pos_keys]
                    set_mocap_position(mocap_body, pos)

            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.01)  
        
        print(f"Finished session: {session_id}")
