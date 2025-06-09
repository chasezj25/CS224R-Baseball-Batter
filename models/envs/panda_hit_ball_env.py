"""
panda_hit_ball_env.py

This module defines a custom OpenAI Gym environment for simulating a 7-DOF Panda robot arm swinging a bat to hit a ball using PyBullet physics. 
The environment provides observations, actions, and rewards suitable for reinforcement learning tasks involving robotic manipulation and dynamic interaction with objects.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class PandaSwingBallEnv(gym.Env):
    """
    Custom Gym environment for simulating a Panda robot arm swinging a bat to hit a baseball using PyBullet.

    This environment is designed for reinforcement learning tasks where the agent controls the 7-DOF Panda arm
    to swing a bat and hit a ball. The environment provides observations including joint angles, bat and ball
    positions, and a flag indicating if the ball has been hit. The reward function encourages hitting the ball
    with speed and distance, and penalizes distance between the bat and ball before hitting.

    Attributes:
        metadata (dict): Rendering modes supported by the environment.
        step_counter (int): Counts the number of steps taken in the current episode.
        max_steps (int): Maximum number of steps per episode.
        time_step (float): Simulation time step.
        physics_client (int): PyBullet physics client ID.
        num_joints (int): Number of controllable joints in the robot.
        ee_link_index (int): Index of the end-effector (bat tip) link.
        action_space (gym.Space): Action space (joint angle deltas).
        observation_space (gym.Space): Observation space (joint angles, bat/ball positions, hit flag).
        ball_hit (bool): Whether the ball has been hit.
        ball_hit_position (np.ndarray or None): Position where the ball was hit.
        prev_bat_pos (np.ndarray or None): Previous bat tip position for speed calculation.
        delta_t (float): Time delta for bat speed calculation.

    Methods:
        __init__():
            Initializes the environment, PyBullet simulation, and spaces.

        _load_env():
            Loads the plane, robot arm with bat, and baseball into the simulation.

        reset(seed=None, options=None):
            Resets the environment to the initial state.
            Returns:
                obs (np.ndarray): Initial observation.
                info (dict): Additional info (empty).

        step(action):
            Applies the action, steps the simulation, computes reward and checks termination.
            Args:
                action (np.ndarray): Joint angle deltas.
            Returns:
                obs (np.ndarray): Next observation.
                reward (float): Reward for the step.
                done (bool): Whether the episode is terminated.
                truncated (bool): Whether the episode is truncated (always False).
                info (dict): Additional info (empty).

        _compute_reward():
            Computes the reward based on bat-ball distance, bat speed, and ball velocity.
            Returns:
                reward (float): Calculated reward.

        _check_termination():
            Checks if the episode should terminate (ball hit ground or max steps).
            Returns:
                done (bool): Termination flag.

        _get_observation():
            Constructs the observation vector (joint angles, bat/ball positions, hit flag).
            Returns:
                obs (np.ndarray): Normalized observation.

        close():
            Disconnects the PyBullet simulation.
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

