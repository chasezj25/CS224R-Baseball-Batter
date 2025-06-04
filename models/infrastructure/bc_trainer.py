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

from models import envs
from models.infrastructure.logger import Logger
from models.infrastructure import utils


class BCTrainer:
    """
    A trainer class for behavior cloning (BC) that handles the training loop,
    data collection, and logging of results. It initializes the environment,
    sets up the agent, and manages the training process.
    """
    def __init__(self, params):
        self.params = params
        self.logger = Logger(params['logdir'])

        # Set random seed
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Make the environment (TODO: build custom environment for pandas robot with bat)
        self.env = gym.make(self.params['env_name'])
        self.env.reset()

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

    def run_training_loop(self, n_iter, collect_policy, eval_policy, 
                          initial_expertdata=None):
        """
        Runs the training loop for the behavior cloning agent. It collects
        trajectories from the environment, adds them to the replay buffer,
        trains the agent, and logs the results. If DAgger is enabled, it will
        also relabel the data with expert actions.
        """

        # Initialize variables for beginning of training
        self.start_time = time.time()

        print("Starting training loop...")
        for iter in range(n_iter):
            # Collect trajectories to be used for training
            paths = self.collect_trajectories(
                iter,
                initial_expertdata,
                collect_policy
            )

            # Add collected data to the replay buffer
            print("Adding paths to replay buffer...")
            self.agent.add_to_replay_buffer(paths)

            # Train the agent using the collected data
            train_info = self.train_agent()

            # Log the training information
            print("Begin Logging...")
            self.perform_logging(
                iter, paths, eval_policy, train_info
            )

            if self.params['save_params']:
                self.agent.save('{}/policy_iter_{}.pt'.format(self.params['logdir'], iter))

    
    def collect_trajectories(self, iter, initial_expertdata, collect_policy=None):
        """
        Collects trajectories from the environment using the specified policy.
        """
        print("Collecting trajectories...")
        if iter > 0 or initial_expertdata is None:
            # Collect trajectories using the specified policy
            paths, _ = utils.sample_trajectories(
                self.env, collect_policy, self.params['num_agent_train_steps'], 
                self.params['ep_len']
            )
        else:
            with open(initial_expertdata, 'rb') as f:
                paths = pickle.load(f)

        return paths

    def train_agent(self):
        """
        Trains the agent using the collected data from the replay buffer.
        """
        print("Training agent...")
        all_logs = []
        for _ in range(self.params['num_agent_train_steps']):

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
        train_returns = [np.sum(np.array(path["rewards"])).real for path in paths]
        eval_returns = [np.sum(eval_path["rewards"]) for eval_path in eval_paths]

        train_ep_lens = [len(path["rewards"]) for path in paths]
        eval_ep_lens = [len(eval_path["rewards"]) for eval_path in eval_paths]

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