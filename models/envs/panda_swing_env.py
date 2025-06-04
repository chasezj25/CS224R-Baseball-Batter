import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

class PandaSwingEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False, demo_trajectory=None):
        super().__init__()
        self.render_mode = render
        self.demo_trajectory = demo_trajectory
        self.bat_offset = np.array([0.0, 0.0, 0.1])
        self.step_counter = 0
        self.max_steps = len(demo_trajectory) - 1 if demo_trajectory is not None else 200
        self.time_step = 1.0 / 100.0 # 100 Hz

        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self._load_env()

        self.num_joints = 7

        # Action: bat tip 6D pose (position + orientation)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(80,), dtype=np.float32)

    def _load_env(self):
        self.plane = p.loadURDF("plane.urdf")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file = os.path.join(current_dir, "panda_arm_bat.urdf")

        self.robot = p.loadURDF(urdf_file, useFixedBase=True)

    def reset(self):
        self.step_counter = 0
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_env()

        if self.demo_trajectory is not None:
            bat_pose = self.demo_trajectory[0][:6]
        else:
            bat_pose = np.zeros(6)

        hand_pos, hand_quat = self._bat_pose_to_hand_ik(bat_pose)

        joint_angles = p.calculateInverseKinematics(self.robot, self.num_joints, hand_pos, hand_quat)
        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, joint_angles[i])

        return self._get_observation()

    def step(self, action):
        self.step_counter += 1
        bat_pose = action[:6]
        hand_pos, hand_quat = self._bat_pose_to_hand_ik(bat_pose)

        joint_angles = p.calculateInverseKinematics(self.robot, self.num_joints, hand_pos, hand_quat)

        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=joint_angles[i])

        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

        obs = self._get_observation()
        done = self.step_counter >= self.max_steps

        reward = 0.0
        if self.demo_trajectory is not None and self.step_counter < len(self.demo_trajectory):
            expert_next = self.demo_trajectory[self.step_counter]
            true_bat_pose = expert_next[:6]
            reward = -np.linalg.norm(action[:3] - true_bat_pose[:3]) \
                     - 0.1 * np.linalg.norm(action[3:6] - true_bat_pose[3:6])

        return obs, reward, done, {}

    def _bat_pose_to_hand_ik(self, bat_pose):
        bat_pos = np.array(bat_pose[:3])
        bat_euler = np.array(bat_pose[3:6])
        bat_quat = p.getQuaternionFromEuler(bat_euler)

        # Rotate offset to get from bat tip to hand
        rot_matrix = np.array(p.getMatrixFromQuaternion(bat_quat)).reshape(3, 3)
        hand_pos = bat_pos - rot_matrix @ self.bat_offset

        return hand_pos.tolist(), bat_quat

    def _get_observation(self):
        if self.demo_trajectory is not None:
            return self.demo_trajectory[self.step_counter].astype(np.float32)
        else:
            return np.zeros(80, dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect()
