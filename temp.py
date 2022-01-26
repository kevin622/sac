import torch
from torch.distributions.normal import Normal
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

env = make_vec_env('Hopper-v2', n_envs=4)
env.reset()
print(env.action_space.sample())
a = np.stack([env.action_space.sample() for i in range(env.num_envs)])
print(a)