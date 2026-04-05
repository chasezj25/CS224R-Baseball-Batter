"""
Runs behavior cloning and DAgger for SwingEnv. Credit to
original authors of homework 1 for the structure of this code.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))  # Add repo root to path

import time
import argparse

from models.agents.bc_agent import BCAgent
from models.infrastructure.bc_trainer import BCTrainer
from models.envs.panda_hit_ball_env import PandaSwingBallEnv


def run_bc(params):
    """
    Sets up agent parameters, initializes BCTrainer, and starts the training loop.
    Prepares the neural network architecture, learning rate, and buffer size, then
    passes them to BCTrainer which manages the full training process.
    """

    #######################
    ## AGENT PARAMS
    #######################

    agent_params = {
        'num_layers': params['num_layers'],
        'size': params['size'],
        'learning_rate': params['learning_rate'],
        'max_replay_buffer_size': params['max_replay_buffer_size'],
    }

    params['agent_class'] = BCAgent
    params['agent_params'] = agent_params

    #######################
    ## RUN TRAINING
    #######################

    trainer = BCTrainer(params)
    trainer.run_training_loop(
        n_iter = params['n_iter'],
        initial_expertdata = params['expert_data'],
        collect_policy = trainer.agent.actor,  
        eval_policy = trainer.agent.actor,
    )


def main():
    """
    Parses command-line arguments, sets up the logging directory, and launches
    the behavior cloning training run.
    """

    parser = argparse.ArgumentParser()
    # NOTE: The file paths are relative to the current working directory
    parser.add_argument('--expert_data', '-ed', type=str, required=True)
    parser.add_argument('--exp_name', '-env', type=str, 
                        default='bc_experiment', required=True)
    parser.add_argument('--env_name', '-env_name', type=str,
                        default='custom/PandaSwingBall-v0', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int, default=200)

    # Sets the number of gradient steps for training policy (per iteration in n_iter)
    parser.add_argument('--num_agent_train_steps', type=int, default=1000)

    # Amount of training data collected (in the env) during each iteration 
    # For final results, recommend a batch size of at least 10,000.
    parser.add_argument('--batch_size', type=int, default=10000)
    # Amount of evaluation data collected (in the env) for logging metrics
    parser.add_argument('--eval_batch_size', type=int, default=10000)
    # Number of sampled data points to be used per gradient/train step
    parser.add_argument('--train_batch_size', type=int, default=100)
    # Number of iterations to run the training loop
    parser.add_argument('--n_iter', type=int, default=1)

    # Depth of the policy to be learned
    parser.add_argument('--num_layers', type=int, default=2)
    # Width of the policy to be learned
    parser.add_argument('--size', type=int, default=64)
    # Learning rate for supervised learning
    parser.add_argument('--learning_rate', type=float, default=5e-3)

    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    
    # Convert arguments to dictionary for easy reference
    params = vars(args)

    # Create the logging directory
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../logs/bc')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    logdir = args.exp_name + '_' + time.strftime('%Y-%m-%d_%H-%M-%S')
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    run_bc(params)


if __name__ == '__main__':
    main()