import pybullet as p
import pybullet_data  # For accessing example URDF files
import random
import time
import os
import math
import pickle

# Initialize PyBullet (replace with your desired initialization)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # If you want to load from pybullet_data

# Load the floor and set gravity
plane_id = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

# Resolve the full path to the URDF file, relative to the script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
urdf_file = os.path.join(current_dir, "panda_arm_bat.urdf")

# Load the URDF file
robot = p.loadURDF(urdf_file, useFixedBase=True)

#load the baseball
baseball_start_pos = [1.0, 0.0, 1.0]
baseball_start_orn = p.getQuaternionFromEuler([0, 0, 0])
baseball_id = p.loadURDF("baseball.urdf", baseball_start_pos, baseball_start_orn)


# Find the end effector link index (usually the last link in the URDF)
end_effector_link_index = p.getNumJoints(robot) - 1

# Get the list of movable joints
movable_joints = []
for i in range(p.getNumJoints(robot)):
    joint_type = p.getJointInfo(robot, i)[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        movable_joints.append(i)

# Load bat swing data
pickle_path = os.path.join(current_dir, "bat_data.pkl") # above directory for most recent bat data
with open(pickle_path, "rb") as f:
    bat_data = pickle.load(f)

# Get the first session swing
first_sess = list(bat_data.keys())[0]
frames = bat_data[first_sess]

start = frames[0]
end = frames[min(10, len(frames) - 1)]

# Direction vector: sweet spot movement over time
swing_vector = [
    end["x"] - start["x"],
    end["y"] - start["y"],
    end["z"] - start["z"]
]

# Normalize direction
norm = math.sqrt(sum(v**2 for v in swing_vector))
swing_dir = [v / norm for v in swing_vector]

# Camera position: behind swing direction
camera_distance = 1.5
camera_target = [
    (start["x"] + end["x"]) / 2,
    (start["y"] + end["y"]) / 2,
    (start["z"] + end["z"]) / 2
]
camera_position = [
    camera_target[i] - swing_dir[i] * camera_distance for i in range(3)
]

# Compute yaw (Z rotation) and pitch (up/down)
yaw = math.degrees(math.atan2(swing_dir[1], swing_dir[0]))
xy_proj = math.sqrt(swing_dir[0]**2 + swing_dir[1]**2)
pitch = -math.degrees(math.atan2(swing_dir[2], xy_proj))

# Apply camera view
p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=yaw,
    cameraPitch=pitch,
    cameraTargetPosition=camera_target
)

# Animate the swing frame by frame
step = 1 / len(frames)
rgb = 1
scale = 1.0 # scale down from human operational space space to robot op space
feet_to_meters = 0.3048
print(frames[0].keys())
x_off = scale * (frames[0]["lajc_x"] + frames[0]["rajc_x"]) / 2
y_off = scale * (frames[0]["lajc_y"] + frames[0]["rajc_y"]) / 2
z_off = scale * (frames[0]["lajc_z"] + frames[0]["rajc_z"]) / 2

for frame in frames:
    # Position (sweet spot)

    pos = [frame["x"] * scale - x_off, frame["y"] * scale - y_off, frame["z"] * scale - z_off]
    """" draw spheres instead of lines, MUCH SLOWER
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=.01,
        rgbaColor= [rgb , 0, 1 - rgb, 1],
    )

)
    p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=pos,
    )
    """
    p.addUserDebugLine(pos, 
                   [pos[0] + 0.01,
                    pos[1],
                    pos[2]],
                   lineColorRGB= [rgb , 0, 1 - rgb],
                   lineWidth=10)
    
    rgb -= step
    # Orientation (converted from degrees to radians, then to quaternion)
    euler = [
        math.radians(frame["x_ang"]),
        math.radians(frame["y_ang"]),
        math.radians(frame["z_ang"])
    ]
    orient = p.getQuaternionFromEuler(euler)

    # Inverse kinematics to compute joint angles
    joint_positions = p.calculateInverseKinematics(
        robot,
        end_effector_link_index,
        targetPosition=pos,
        targetOrientation=orient
    )

    # Apply IK output to movable joints only
    for joint_idx, target_pos in zip(movable_joints, joint_positions):
        p.setJointMotorControl2(
            bodyIndex=robot,
            jointIndex=joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos,
            force=200, # seems like default value is 500
            positionGain=0.1
        )

    p.applyExternalForce(objectUniqueId=baseball_id,
                     linkIndex=-1,  
                     forceObj=[0,0, 9.81 *0.1 ], # note mass = 0.1
                     posObj=[0,0,0],
                     flags=p.LINK_FRAME)

    p.stepSimulation()
    time.sleep(0.05)  # Assuming 100Hz data frequency
    #time.sleep(0.0028)

# === Keep simulation window open ===
while True:
    p.stepSimulation()
    time.sleep(0.01)
    