"""
panda_hit_ball_env.py

This module defines a custom OpenAI Gym environment for simulating a 7-DOF Panda robot arm swinging a bat to hit a ball using PyBullet physics. 
The environment provides observations, actions, and rewards suitable for reinforcement learning tasks involving robotic manipulation and dynamic interaction with objects.
"""

import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os

class PandaSwingBallEnv(gym.Env):
    """
    Custom Gym environment for a Panda robot arm swinging a bat to hit a ball.
    The environment uses PyBullet for physics simulation.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
        """
        Initialize the environment.
        """
        super().__init__()
        self.render_mode = render
        self.step_counter = 0
        self.max_steps = 300
        self.time_step = 1.0 / 100.0

        # Connect to PyBullet physics server
        if self.render_mode:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self._load_env()

        self.num_joints = 7
        self.ee_link_index = 7 # End-effector link index (adjust if different)

        # Action space: 7 joint angle targets for the arm
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: joint positions (7), joint velocities (7), ball position (3), ball velocity (3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.ball_hit = False
        self.ball_hit_position = None

    def _load_env(self):
        """
        Load the simulation environment: plane, robot arm, and ball.
        """
        self.plane = p.loadURDF("plane.urdf")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_file = os.path.join(current_dir, "panda_arm_bat.urdf")
        self.robot = p.loadURDF(urdf_file, useFixedBase=True)

        ball_start_pos = [1.0, 0, 1.0]
        ball_start_orn = p.getQuaternionFromEuler([0, 0, 0])
        self.ball = p.loadURDF("baseball.urdf", ball_start_pos, ball_start_orn)

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.step_counter = 0
        self.ball_hit = False
        self.ball_hit_position = None
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._load_env()

        # Reset all robot joints to zero position
        for i in range(self.num_joints):
            p.resetJointState(self.robot, i, 0.0)

        return self._get_observation()

    def step(self, action):
        """
        Apply an action to the environment and advance the simulation.
        """
        self.step_counter += 1

        # Apply joint position control
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=action[i])

        # Step the simulation forward
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode:
                time.sleep(self.time_step)

        # If the ball has not been hit, reset its position and velocity
        if self.ball_hit:
            pass  # Ball already hit, let it move naturally
        else:
            p.resetBaseVelocity(self.ball, linearVelocity=[0.0, 0.0, 0.0])
            p.resetBasePositionAndOrientation(self.ball, [0.7, 0, 1.0], [0, 0, 0, 1])

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_termination()

        return obs, reward, done

    def _compute_reward(self):
        """
        Compute the reward for the current state.
        """
        bat_tip_pos = p.getLinkState(self.robot, self.ee_link_index)[0]
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_vel, _ = p.getBaseVelocity(self.ball)

        bat_speed = np.linalg.norm(p.getLinkState(self.robot, self.ee_link_index, computeLinkVelocity=1)[6])
        distance = np.linalg.norm(np.array(bat_tip_pos) - np.array(ball_pos))

        HIT_REWARD = 100.0
        DISTANCE_WEIGHT = -10.0
        BAT_SPEED_WEIGHT = 1.0
        BALL_VELOCITY_WEIGHT = 2.0

        # Reward for hitting the ball, otherwise encourage bat speed and proximity
        if not self.ball_hit:
            if distance < 0.1:
                self.ball_hit = True
                self.ball_hit_position = ball_pos
                return HIT_REWARD + BALL_VELOCITY_WEIGHT * np.linalg.norm(ball_vel) + BAT_SPEED_WEIGHT * bat_speed
            else:
                return DISTANCE_WEIGHT * distance + BAT_SPEED_WEIGHT * bat_speed
        else:
            return BALL_VELOCITY_WEIGHT * np.linalg.norm(ball_vel)

    def _check_termination(self):
        """
        Check if the episode should terminate.
        """
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)

        # Terminate if ball has been hit and falls below a certain height
        if self.ball_hit and ball_pos[2] < 0.1:
            return True  # Ball hit ground

        # Terminate if maximum steps reached
        if self.step_counter >= self.max_steps:
            return True

        return False

    def _get_observation(self):
        """
        Get the current observation of the environment.
        """
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        joint_positions = [s[0] for s in joint_states]
        joint_velocities = [s[1] for s in joint_states]

        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)
        ball_vel, _ = p.getBaseVelocity(self.ball)

        return np.array(joint_positions + joint_velocities + list(ball_pos) + list(ball_vel), dtype=np.float32)

    def render(self, mode="human"):
        """
        Render the environment. (Not implemented)
        """
        pass

    def close(self):
        """
        Clean up and disconnect from the physics server.
        """
        p.disconnect()