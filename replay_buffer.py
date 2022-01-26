# random policy data generation
import numpy as np
import torch
from torch.optim import Adam
import gym
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
from collections import deque
import sys
import random
# sys.path.append("/home/kookhyun/sac/models")
# from models import Policy

# parallel environments
n_envs = 4
env = make_vec_env("Hopper-v2", n_envs=n_envs)

# initial dataset
replay_buffer = []
buffer_size = 10

# random policy
observations = env.reset()
state_shape = observations.shape[1]
action_shape = env.action_space.shape[0]

# policy = Policy(input_size=state_shape, output_size=action_shape)


class ReplayBuffer:

    def __init__(self, size, device, env):
        self.size = size
        self.device = device
        self.env = env

        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        self.memory = [
            torch.zeros((size, state_shape)).to(device),
            torch.zeros((
                size,
                action_shape,
            )).to(device),
            torch.zeros((size, 1)).to(device),
            torch.zeros((
                size,
                state_shape,
            )).to(device),
            torch.zeros((size, 1)).to(device)
        ]  # s, a, r, s_prime, done
        self.change_idx = 0

    def __len__(self):
        return self.size

    def append(self, content):
        '''
        content : s, a, r, s_prime, done
        '''

        for i, c in enumerate(content):
            self.memory[i][self.change_idx] = torch.tensor(c).to(self.device)

        # Update next location to change
        self.change_idx += 1
        if self.change_idx == self.size:
            self.change_idx = 0

    # def generate_data(self, policy):
    def generate_data(self):
        env = self.env
        observations = env.reset()
        n_envs = env.num_envs
        print('Generating Dataset')
        for _ in tqdm(range(self.size // n_envs)):
            # actions = torch.tanh(
            #     policy(torch.from_numpy(observations).to(torch.float32).to(
            #         self.device)).sample()).to("cpu").numpy()
            actions = np.stack([env.action_space.sample() for _ in range(n_envs)])
            new_observations, rewards, dones, infos = env.step(actions)
            for i in range(n_envs):
                # s, a, r, s_prime, done
                self.append([observations[i], actions[i], rewards[i], new_observations[i], dones[i]])
            observations = new_observations
        print('Done!')

    def random_sample(self, size):
        indices = random.sample(range(len(self)), k=size)
        states = self.memory[0][indices].to(self.device)
        actions = self.memory[1][indices].to(self.device)
        rewards = self.memory[2][indices].to(self.device)
        state_primes = self.memory[3][indices].to(self.device)
        dones = self.memory[4][indices].to(self.device)
        return states, actions, rewards, state_primes, dones


# replay_buffer = ReplayBuffer(maxlen=100)
# replay_buffer.generate_data(policy=policy, env=env, n_envs=env.num_envs)
