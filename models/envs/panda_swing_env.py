import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class PandaSwingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = render # Rendering for debugging
        self.time_step = 1.0 / 240.0 # Frame rate for Sampling
        self.max_steps = 200 # Maximum steps taken per episode
        self.step_counter = 0 # Counter for steps taken in the current episode

        # Start PyBullet simulation
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For loading URDFs
        p.setGravity(0, -9.81, 0) # Set gravity (y-axis up environment) 

        self._load_env() # Load the environment

        self.num_joints = p.getNumJoints(self.robot) # Number of joints in the robot
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)  # Joint angle deltas 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32) #  

    def _load_env(self):
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        self.bat = p.loadURDF("bat/bat.urdf", useFixedBase=False)

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_env()

        # Neutral position
        joint_positions = [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
        for i in range(7):
            p.resetJointState(self.robot, i, joint_positions[i])

        return self._get_observation()

    def step(self, action):
        self.step_counter += 1

        # Apply joint deltas
        for i in range(7):
            current = p.getJointState(self.robot, i)[0]
            target = current + action[i]
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=target)

        for _ in range(10):  # simulate real time
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

        obs = self._get_observation()
        reward = -np.linalg.norm(obs[:3])  # e.g. keep wrist near origin
        done = self.step_counter >= self.max_steps
        return obs, reward, done, {}

    def _get_observation(self):
        joint_states = [p.getJointState(self.robot, i)[:2] for i in range(7)]
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]
        return np.array(joint_positions + joint_velocities, dtype=np.float32)

    def render(self, mode="human"):
        pass  # PyBullet GUI handles rendering

    def close(self):
        p.disconnect()