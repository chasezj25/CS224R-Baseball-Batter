"""
Defines the BCTrainer class for training a behavior cloning model. Adapted 
from the original code in homework 1.
"""

from collections import OrderedDict

import pickle
import time
import torch
import gym
import numpy as np

from models.infrastructure import torch_utils as ptu
from models.infrastructure.logger import Logger
from models.infrastructure import utils


class BCTrainer:
    """

    """
    def __init__(self, params):
        self.params = params
        self.logger = Logger(params['logdir'])

        # Set random seed
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Make the environment (TODO: build custom environment for pandas robot with bat)
        self.env = None
        self.env.reset(seed=seed)

        # Set the maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        # Make sure the environment is continuous and not discrete
        assert not isinstance(self.env.action_space, gym.spaces.Discrete), \
            "Environment must have a continuous action space."
        
        # Set observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ob_dim'] = ob_dim
        self.params['agent_params']['ac_dim'] = ac_dim

        # Initialize the agent
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, collect_policy, eval_policy, 
                          initial_expertdata=None):
        """
        """

        # Initialize variables for beginning of training
        self.start_time = time.time()

        print("Starting training loop...")
        # Collect trajectories to be used for training
        paths, envsteps_this_batch = self.collect_trajectories(
            initial_expertdata,
            collect_policy,
        )

        # Add collected data to the replay buffer
        print("Adding paths to replay buffer...")
        self.agent.replay_buffer.add_paths(paths)

        # Train the agent using the collected data
        train_info = self.train_agent()

        # Log the training information
        print("Begin Logging...")
        self.perform_logging(
            iter, paths, eval_policy, train_info
        )

        if self.params['save_params']:
            self.agent.save('{}/policy.pt'.format(self.params['logdir'], iter))

    
    def collect_trajectories(self, iter, initial_expertdata, collect_policy):
        """
        Collects trajectories from the environment using the specified policy.
        """
        print("Collecting trajectories...")
        
        with open(initial_expertdata, 'rb') as f:
            paths = pickle.load(f)

        return paths

    def train_agent(self):
        """
        """
        print("Training agent...")
        all_logs = []
        for _ in range(self.params['num_agent_train_steps_per_iter']):

            # Sample from replay buffer
            ob_batch, ac_batch, reward_batch, next_ob_batch, terminal_batch = self.agent.sample(
                self.params['train_batch_size'])
            
            # Train the agent with the sampled batch
            train_info = self.agent.train(ob_batch, ac_batch)
            all_logs.append(train_info)
        
        return all_logs

    def perform_logging(self, iter, paths, eval_policy, train_info):
        """
        Logs the training information and evaluation metrics.
        """
        print("Performing logging...")
        
        # Collect evaluation paths for logging
        eval_paths, _ = utils.sample_trajectories(
            self.env, eval_policy, self.params['eval_batch_size'], 
            self.params['ep_len']
        )

        # Save evaluation metrics
        # Get the returns and episode lengths of all paths, for logging
        train_returns = [path["reward"].sum() for path in paths]
        eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

        train_ep_lens = [len(path["reward"]) for path in paths]
        eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

        # Define logged metrics
        logs = OrderedDict()
        logs["Eval_AverageReturn"] = np.mean(eval_returns)
        logs["Eval_StdReturn"] = np.std(eval_returns)
        logs["Eval_MaxReturn"] = np.max(eval_returns)
        logs["Eval_MinReturn"] = np.min(eval_returns)
        logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

        logs["Train_AverageReturn"] = np.mean(train_returns)
        logs["Train_StdReturn"] = np.std(train_returns)
        logs["Train_MaxReturn"] = np.max(train_returns)
        logs["Train_MinReturn"] = np.min(train_returns)
        logs["Train_AverageEpLen"] = np.mean(train_ep_lens)
        
        logs['TimeSinceStart'] = time.time() - self.start_time
        last_log = train_info[-1]
        logs.update(last_log)

        self.initial_return = np.mean(train_returns)
        logs['Initial_DataCollection_AverageReturn'] = self.initial_return

        # Perform logging with tensorboard
        for key, value in logs.items():
            print(f"{key}: {value}")
            self.logger.log_scalar(value, key, iter)
        print("Logging complete...\n\n")

        self.logger.flush()