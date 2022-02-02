import os
import random
import pickle

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env


class ReplayBuffer:
    def __init__(self, device, args):
        random.seed(args.seed)
        self.device = device
        self.env_name = args.env_name
        self.capacity = args.buffer_size
        self.memory = []
        self.change_idx = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done):
        '''
        content : state, action, reward, next_state, done
        '''
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.change_idx] = [state, action, reward, next_state, done]

        # Update next location to change
        self.change_idx += 1
        if self.change_idx == self.capacity:
            self.change_idx = 0

    def random_sample(self, size):
        samples = random.sample(self.memory, k=size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*samples))
        return states, actions, rewards, next_states, dones
    
    def save(self):
        if not os.path.exists("checkpoints/"):
            os.makedirs('checkpoints/')
        
        save_path = f'checkpoints/buffer_{self.env_name}'
        print(f'Saving Memory in {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)

    def load(self):
        save_path = f'checkpoints/buffer_{self.env_name}'
        print(f'Loading existing buffer from {save_path}')
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)
            self.change_idx = 0
            print('Loading complete!')
