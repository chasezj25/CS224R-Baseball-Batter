"""
A simple implementation of a Behavior Cloning (BC) agent using PyTorch.
Adapted from original code in homework 1.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch import distributions

import numpy as np

from models.infrastructure import utils
from models.infrastructure.replay_buffer import ReplayBuffer

def build_mlp(input_dim, output_dim, n_layers=2, hidden_dim=64):
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.ReLU())
    
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)

class BCPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=2, hidden_dim=64, learning_rate=1e-3):
        """
        Initialize the BCPolicy neural network agent.
        """
        super(BCPolicy, self).__init__()

        # Initialize the neural network
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        
        self.network = build_mlp(
            input_dim=self.input_dim, 
            output_dim=self.output_dim, 
            n_layers=self.n_layers, 
            hidden_dim=self.hidden_dim
        )

        self.network.to(utils.device)
        self.logstd = nn.Parameter(torch.zeros(self.output_dim, dtype=torch.float32, device=utils.device))
        self.logstd.to(utils.device)
        self.optimizer = optim.Adam(
            list(self.network.parameters()) + [self.logstd],
            lr=self.learning_rate
        )

    def save(self, path):
        """
        Save the model to the specified path.
        """
        torch.save(self.state_dict(), path)

    def forward(self, observation):
        """
        Forward pass through the network to get the predicted action distribution.
        """
        mean = self.network(observation)
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mean, std)
        return dist

    def update(self, observation, action):
        """
        Update the policy using mean squared error loss between sampled action and target actions.
        """
        observation = utils.from_numpy(observation.astype(np.float32))
        action = utils.from_numpy(action.astype(np.float32))

        # Get predicted action distribution and sample from it
        dist = self.forward(observation)
        pred_action = dist.rsample()
        loss = F.mse_loss(pred_action, action)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'Training Loss': loss.to('cpu').detach().numpy()
        }

    def get_action(self, observation):
        """
        Get the predicted action for the given observation by sampling from the distribution.
        """
        if len(observation.shape) <= 1:
            observation = observation[None]

        observation = utils.from_numpy(observation.astype(np.float32))
        dist = self.forward(observation)
        action = dist.rsample()
        return utils.to_numpy(action)

class BCAgent:
    """
    A Behavior Cloning agent that uses a neural network policy to mimic expert behavior.
    """
    def __init__(self, env, agent_params):
        self.env = env
        self.agent_params = agent_params
        self.actor = BCPolicy(
            input_dim=agent_params['ob_dim'],
            output_dim=agent_params['ac_dim'],
            n_layers=agent_params['num_layers'],
            hidden_dim=agent_params['size'],
            learning_rate=agent_params['learning_rate']
        )

        self.replay_buffer = ReplayBuffer(
            self.agent_params['max_replay_buffer_size'],
        )

    def train(self, observation, action):
        log = self.actor.update(observation, action)
        return log
    
    def add_to_replay_buffer(self, paths):
        """Add a list of trajectory rollouts to the replay buffer."""
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        """
        Sample a batch of data from the replay buffer.
        """
        return self.replay_buffer.sample_random_data(batch_size)
    
    def save(self, path):
        """
        Save the agent's model to the specified path.
        """
        return self.actor.save(path)