import argparse
import sys
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    # Evaluation settings
    parser = argparse.ArgumentParser(description="RL - Lorenz '63")

    parser.add_argument('--log_dir', type=str, required=True,
                        help='Directory to store Log Files')
    parser.add_argument('--neurons_perLayer', type=int, required=True, default=512, help="Neurons per hidden layer for the actor and critic networks")
    parser.add_argument('--gamma', type=float, required=True, default=1.0001, help="Learning Rate")
    parser.add_argument('--learning_rate', type=float, required=True, default=0.00002, help="Learning Rate")
    parser.add_argument('--batchSize', type=int, required=True, default=256, help="Batch Size")
    parser.add_argument('--total_timesteps', type=float, required=True, default=5e6, help="Total number of timesteps to train the RL agent")
    parser.add_argument('--vf_coeff', type=float, required=True, default=0.5, help="value function coefficient")
    parser.add_argument('--max_grad_norm', type=float, required=True, default=0.85, help="gradient clipping coefficient")

    parser.add_argument('--assim_freq', type=int, required=True, default=24, help="Assimilation Frequency, every this many steps we assimilate")
    parser.add_argument('--obs_Noise', type=float, required=True, default=1., help="Assimilation Observational Noise")    
    
    parser.add_argument('--device', required=True, help='device')         

    args = parser.parse_args()

    return args







