import os
import random

import numpy as np
import torch
# from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, size, device):
        self.size = size
        self.device = device
        # self.env = env

        # state_shape = env.observation_space.shape[0]
        # action_shape = env.action_space.shape[0]
        # self.memory = [
        #     torch.zeros((size, state_shape)).to(device),
        #     torch.zeros((
        #         size,
        #         action_shape,
        #     )).to(device),
        #     torch.zeros((size, 1)).to(device),
        #     torch.zeros((
        #         size,
        #         state_shape,
        #     )).to(device),
        #     torch.zeros((size, 1)).to(device)
        # ]  # s, a, r, s_prime, done
        self.memory = []
        self.change_idx = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done):
        '''
        content : s, a, r, s_prime, done
        '''
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.change_idx] = (state, action, reward, next_state, done)
        # Update next location to change
        self.change_idx += 1
        if self.change_idx == self.size:
            self.change_idx = 0
        # for i, c in enumerate(content):
        #     self.memory[i][self.change_idx] = torch.tensor(c).to(self.device)


    # def generate_data(self, policy):
    # def generate_data(self):
    #     if os.path.exists('checkpoints/replay_buffer.pt'):
    #         self.load()
    #     else:
    #         env = self.env
    #         observations = env.reset()
    #         n_envs = env.num_envs
    #         print('Generating Dataset')
    #         for _ in tqdm(range(self.size // n_envs)):
    #             actions = np.stack([env.action_space.sample() for _ in range(n_envs)])
    #             new_observations, rewards, dones, infos = env.step(actions)
    #             for i in range(n_envs):
    #                 # s, a, r, s_prime, done
    #                 self.append([observations[i], actions[i], rewards[i], new_observations[i], dones[i]])
    #             observations = new_observations
    #         print('Done!')
    #         self.save()

    def random_sample(self, size):
        indices = random.sample(range(len(self)), k=size)
        # states = self.memory[0][indices].to(self.device)
        # actions = self.memory[1][indices].to(self.device)
        # rewards = self.memory[2][indices].to(self.device)
        # next_states = self.memory[3][indices].to(self.device)
        # dones = self.memory[4][indices].to(self.device)
        batch = random.sample(self.memory, size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    # def save(self):
    #     if not os.path.exists("checkpoints/"):
    #         os.makedirs('checkpoints/')
    #     torch.save(self.memory, 'checkpoints/replay_buffer.pt')
    #     print('Saving Memory in checkpoints/replay_buffer.pt')

    # def load(self):
    #     print('Loading existing data from checkpoints/replay_buffer.pt')
    #     self.memory = torch.load('checkpoints/replay_buffer.pt', map_location=self.device)
    #     print('Loading complete!')
    #     self.change_idx = 0
