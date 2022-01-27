import random
import numpy as np


class ReplayBuffer:
    def __init__(self, size, device):
        self.size = size
        self.device = device
        self.memory = []
        self.change_idx = 0

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, mask):
        '''
        content : s, a, r, s_prime, mask
        '''
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.change_idx] = (state, action, reward, next_state, mask)
        # Update next location to change
        self.change_idx += 1
        if self.change_idx == self.size:
            self.change_idx = 0

    def random_sample(self, size):
        batch = random.sample(self.memory, size)
        states, actions, rewards, next_states, masks = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, masks

