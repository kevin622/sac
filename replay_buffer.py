import os
import random

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, device, env, args):
        self.device = device
        self.size = args.buffer_size
        self.env = env
        self.env_name = args.env_name
        
        random.seed(args.seed)
        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        self.memory = [
            torch.zeros((self.size, state_shape)).to(device),
            torch.zeros((
                self.size,
                action_shape,
            )).to(device),
            torch.zeros((self.size, 1)).to(device),
            torch.zeros((
                self.size,
                state_shape,
            )).to(device),
            torch.zeros((self.size, 1)).to(device)
        ]  # s, a, r, s_prime, done
        self.change_idx = 0

    def __len__(self):
        return len(self.memory[0])

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
        if os.path.exists(f'checkpoints/replay_buffer_{self.env_name}_{self.size}.pt'):
            self.load()
        else:
            env = self.env
            observations = env.reset()
            n_envs = env.num_envs
            print('Generating Dataset')
            for _ in tqdm(range(self.size // n_envs)):
                actions = np.stack([env.action_space.sample() for _ in range(n_envs)])
                new_observations, rewards, dones, infos = env.step(actions)
                for i in range(n_envs):
                    # s, a, r, s_prime, done
                    self.append([observations[i], actions[i], rewards[i], new_observations[i], dones[i]])
                observations = new_observations
            print('Done!')
            self.save()

    def random_sample(self, size):
        indices = random.sample(range(len(self)), k=size)
        states = self.memory[0][indices].detach().to(self.device)
        actions = self.memory[1][indices].detach().to(self.device)
        rewards = self.memory[2][indices].detach().to(self.device)
        next_states = self.memory[3][indices].detach().to(self.device)
        dones = self.memory[4][indices].detach().to(self.device)
        return states, actions, rewards, next_states, dones
    
    def save(self):
        if not os.path.exists("checkpoints/"):
            os.makedirs('checkpoints/')
        torch.save(self.memory, f'checkpoints/replay_buffer_{self.env_name}_{self.size}.pt')
        print('Saving Memory in checkpoints/replay_buffer.pt')

    def load(self):
        print(f'Loading existing data from checkpoints/replay_buffer_{self.env_name}_{self.size}.pt')
        self.memory = torch.load(f'checkpoints/replay_buffer_{self.env_name}_{self.size}.pt', map_location=self.device)
        print('Loading complete!')
        print(f'Tensor size {len(self.memory[0])}')
        self.change_idx = 0
