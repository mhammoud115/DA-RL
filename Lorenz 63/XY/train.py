import gym
import numpy as np
import torch

from stable_baselines3 import SAC, PPO, HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
import gym

from newLorenzEnv import Lorenz63Env
from stable_baselines3 import PPO
import time 
import matplotlib.pyplot as plt
from typing import Callable

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from stable_baselines3.common.evaluation import evaluate_policy



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value + 0.00001

    return func


def oneStep_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        
        if progress_remaining<0.5:
            return initial_value
        else:
            return initial_value*0.1

    return func

def make_env(env_id, rank, integ_step, noise, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Lorenz63Env(integ_steps = integ_step)
        env.seed(seed + rank)
        return env
    set_random_seed(seed+ rank)
    return _init


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(self.save_path)
        return True
  

def main(args):
    print(args)
    print('torch version: ', torch.__version__)
    
    device = torch.device(args.device)
    os.makedirs(args.log_dir, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    num_cpu = 70  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env('Lorenz', rank=i, integ_step=int(args.assim_freq), noise=args.obs_Noise) for i in range(num_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(activation_fn=torch.nn.Softplus, 
                            net_arch=dict(pi=[args.neurons_perLayer, args.neurons_perLayer], vf=[args.neurons_perLayer, args.neurons_perLayer]))
    

    model = PPO("MlpPolicy",
                env,
                # policy_kwargs=policy_kwargs,
                n_steps=args.batchSize,
                batch_size=args.batchSize,
                n_epochs=20,
                learning_rate=args.learning_rate,
                gamma=args.gamma,
                ent_coef=0.0,
                vf_coef=args.vf_coeff,
                max_grad_norm=args.max_grad_norm,
                verbose=1,
                tensorboard_log=args.log_dir)

    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=args.log_dir)
    
    model.learn(total_timesteps=args.total_timesteps, log_interval=100, progress_bar=False, tb_log_name="PPO_MLP_L63", callback=callback)
    model.save(args.log_dir+"PPO_Lorenz63")



if __name__ == '__main__':
    from opts import parse_args
    args = parse_args()
    main(args)



