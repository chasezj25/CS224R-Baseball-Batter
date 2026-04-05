"""
panda_hit_ball_env.py

Custom Gym environment for a 7-DOF Panda robot arm swinging a bat to hit a ball
using PyBullet physics. Provides observations, actions, and rewards for RL training.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class PandaSwingBallEnv(gym.Env):
    """
    Gym environment where a Panda robot arm swings a bat to hit a baseball.

    Observations include joint angles, bat tip position, ball position, and a
    binary flag for whether the ball has been hit. The reward encourages fast,
    accurate contact and penalises the distance between bat and ball before contact.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.step_counter = 0
        self.max_steps = 200
        self.time_step = 1.0 / 100.0

        self.render_mode = render_mode
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.num_joints = 7
        self.ee_link_index = 7

        self._load_env()

        # Action space: joint angle deltas
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: joint angles (7) + bat pos (3) + ball pos (3) + ball_hit (1) = 14
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(14,), dtype=np.float32)

        self.ball_hit = False
        self.ball_hit_position = None
        self.prev_bat_pos = None
        self.delta_t = 0.01

    def _load_env(self):
        self.plane = p.loadURDF("plane.urdf")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file = os.path.join(current_dir, "panda_arm_bat.urdf")

        # Add small random noise to the initial joint positions for the bat/arm
        initial_joint_positions = np.zeros(self.num_joints)
        noise = np.random.uniform(-0.05, 0.05, size=self.num_joints)
        initial_joint_positions += noise

        self.robot = p.loadURDF(urdf_file, useFixedBase=True)

        # Set the initial joint positions with noise
        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, initial_joint_positions[i])

        ball_start_pos = [0.0, -1.0, 0.7]
        baseball_urdf = os.path.join(current_dir, "baseball.urdf")
        self.ball = p.loadURDF(baseball_urdf, ball_start_pos)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_counter = 0
        self.ball_hit = False
        self.ball_hit_position = None
        self.prev_bat_pos = None

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_env()

        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, 0.0)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        self.step_counter += 1

        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        current_angles = np.array([s[0] for s in joint_states])
        new_angles = current_angles + action

        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=new_angles[i])

        for _ in range(10):
            p.applyExternalForce(objectUniqueId=self.ball,
                                 linkIndex=-1,
                                 forceObj=[0, 0, 9.81 * 0.1],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME)
            p.stepSimulation()

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()
        info = {}

        return obs, reward, done, False, info  # match gymnasium step() return

    def _compute_reward(self):
        # Get current bat tip position
        bat_tip_pos = p.getLinkState(self.robot, self.ee_link_index)[0]

        # Get ball position and velocity
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_vel, _ = p.getBaseVelocity(self.ball)

        # Compute bat tip linear speed
        if self.prev_bat_pos is None:
            bat_speed = 0.0
        else:
            bat_speed = np.linalg.norm((np.array(bat_tip_pos) - np.array(self.prev_bat_pos)) / self.delta_t)

        self.prev_bat_pos = np.array(bat_tip_pos)

        # Distance between bat tip and ball
        distance = np.linalg.norm(np.array(bat_tip_pos) - np.array(ball_pos))

        # Rotation encouragement — encourage counter-clockwise rotation around Z
        link_state = p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=1)
        bat_angular_vel = np.array(link_state[7])  # 8th element is local angular velocity
        LEFT_ROTATION_WEIGHT = 1.0
        left_rotation_reward = LEFT_ROTATION_WEIGHT * bat_angular_vel[2]  # Z component

        # Tuned weights — lower std, smoother reward
        HIT_REWARD = 100.0
        DISTANCE_WEIGHT = -2.0
        BAT_SPEED_WEIGHT = 0.5
        BALL_VELOCITY_WEIGHT = 1.0
        BALL_X_DISTANCE_REWARD_WEIGHT = 1.0  # Use only X distance now

        # Compute reward
        if not self.ball_hit:
            if distance < 0.1:
                self.ball_hit = True
                self.ball_hit_position = ball_pos
                return (
                    HIT_REWARD
                    + BALL_VELOCITY_WEIGHT * np.linalg.norm(ball_vel)
                    + BAT_SPEED_WEIGHT * bat_speed
                    + left_rotation_reward
                )
            else:
                return (
                    DISTANCE_WEIGHT * distance
                    + BAT_SPEED_WEIGHT * bat_speed
                    + left_rotation_reward
                )
        else:
            # After hit — encourage ball moving far along X
            ball_x_distance = abs(ball_pos[0])  # Or just ball_pos[0] if you want to prefer positive X only
            return (
                BALL_VELOCITY_WEIGHT * np.linalg.norm(ball_vel)
                + BALL_X_DISTANCE_REWARD_WEIGHT * ball_x_distance
                + left_rotation_reward
            )


    def _check_termination(self):
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        if self.ball_hit and ball_pos[2] < 0.05:
            return True
        if self.step_counter >= self.max_steps:
            return True
        return False

    def _get_observation(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_positions = [s[0] for s in joint_states]

        bat_pos = p.getLinkState(self.robot, self.ee_link_index)[0]
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)

        ball_hit_flag = 1.0 if self.ball_hit else 0.0

        obs = np.array(joint_positions + list(bat_pos) + list(ball_pos) + [ball_hit_flag], dtype=np.float32)

        # Normalize observation to [-1, 1] (temporary simple scaling)
        obs = np.clip(obs / 10.0, -1.0, 1.0)

        return obs

    def close(self):
        try:
            if p.getConnectionInfo()['isConnected']:
                p.disconnect()
        except Exception as e:
            print(f"[WARNING] Exception in close(): {e}")

