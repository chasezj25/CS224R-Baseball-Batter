import mujoco
import os
import mujoco.viewer

# Path to the humanoid model XML file (update if needed)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'humanoid_mocap_model.xml')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Could not find humanoid.xml at {MODEL_PATH}")

# Load the model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Launch the viewer (no simulation, just viewing)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        pass  # Keeps the viewer open until the window is closed
    print("Close the window to exit.")