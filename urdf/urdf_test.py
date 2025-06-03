import pybullet as p
import pybullet_data  # For accessing example URDF files

# Initialize PyBullet (replace with your desired initialization)
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) # If you want to load from pybullet_data

# Path to your URDF file
urdf_file = "panda_arm_bat.urdf"  # Or the full path if not in the search path
# Load the URDF file
robot = p.loadURDF(urdf_file, useFixedBase=True)

while True:
    pass