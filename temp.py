import torch
from torch.nn import Linear
from models.models import Policy
import gym

a = gym.make("Hopper-v2")
print(dir(a))
print(a._max_episode_steps)