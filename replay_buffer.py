import os
import random
import pickle

import numpy as np

class ReplayBuffer:
    def __init__(self, seed, buffer_size, env_name):
        random.seed(seed)
        self.capacity = buffer_size
        self.env_name = env_name
        self.memory = []
        self.change_idx = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, mask):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.change_idx] = (state, action, reward, next_state, mask)

        # Update next location to change
        self.change_idx = (self.change_idx + 1) % self.capacity

    def sample(self, size):
        samples = random.sample(self.memory, k=size)
        states, actions, rewards, next_states, masks = map(np.stack, zip(*samples))
        return states, actions, rewards, next_states, masks
    
    def save(self):
        if not os.path.exists("checkpoints/"):
            os.makedirs('checkpoints/')
        
        save_path = f'checkpoints/buffer_{self.env_name}'
        idx_save_path = f'checkpoints/buffer_{self.env_name}_idx'
        print(f'Saving Memory in {save_path}')

        with open(save_path, 'wb') as f:
            pickle.dump(self.memory, f)
        with open(idx_save_path, 'wb') as f:
            pickle.dump(self.change_idx, f)

    def load(self):
        save_path = f'checkpoints/buffer_{self.env_name}'
        idx_save_path = f'checkpoints/buffer_{self.env_name}_idx'
        print(f'Loading existing buffer from {save_path}')
        with open(save_path, 'rb') as f:
            self.memory = pickle.load(f)
            print('Loading complete!')
        with open(idx_save_path, 'rb') as f:
            self.change_idx = pickle.load(f)
