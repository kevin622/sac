import os
import random
import pickle

import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        random.seed(args.seed)
        self.capacity = args.buffer_size
        self.env_name = args.env_name
        self.memory = []
        self.change_idx = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, mask):
        '''
        content : state, action, reward, next_state, mask
        '''
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
