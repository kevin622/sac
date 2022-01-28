from typing import OrderedDict
import torch
from torch.optim import Adam
import torch.nn.functional  as F
from models import Policy, QNetwork, ValueNetwork
from data_generation.data_generation import ReplayBuffer
from utils import to_numpy, to_tensor

class SAC(object):
    def __init__(self, device, args, env):
        # hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.device = device
        state_shape = env.observation_space.shape[0]
        action_shape = env.action_space.shape[0]
        
        # Neural Nets
        # Policy
        self.policy = Policy(input_size=state_shape, output_size=action_shape).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        # Q values
        self.q_network = QNetwork(input_size=(state_shape + action_shape)).to(device)
        self.q_optim = Adam(self.q_network.parameters(), lr=args.lr)
        # Value
        self.value_network = ValueNetwork(input_size=state_shape).to(device)
        self.value_optim = Adam(self.value_network.parameters(), lr=args.lr)
        # Value Exponential Average
        self.value_avg_network = ValueNetwork(input_size=state_shape).to(device)
        for target_parm, param in zip(self.value_avg_network.parameters(), self.value_network.parameters()):
            target_parm.data.copy_(param.data)

    def get_action(self, state):
        action = self.policy.sample(state)[0]
        return to_numpy(action)
        
    
    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        '''
        returns (value_loss, Q1_loss, Q2_loss, policy_loss)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.random_sample(size=batch_size)
        
        # Update Value
        with torch.no_grad():
            curr_policy_action_batch, curr_policy_log_prob, _ = self.policy.sample(state_batch)
            curr_policy_Q1_st_at, curr_policy_Q2_st_at = self.q_network(torch.cat((state_batch, curr_policy_action_batch), dim=-1))
            curr_policy_Q_st_at = torch.min(curr_policy_Q1_st_at, curr_policy_Q2_st_at)
        V_st = self.value_network(state_batch)
        value_loss = F.mse_loss(V_st, curr_policy_Q_st_at - self.alpha * curr_policy_log_prob)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        
        # Update Q
        # Use the exponenetial average parameters for Value network
        with torch.no_grad():
            V_st_1 = self.value_avg_network(next_state_batch)
            mask_batch = done_batch * (-1) + 1
            Q_hat_st_at = reward_batch + mask_batch * self.gamma * V_st_1
        Q1_st_at, Q2_st_at = self.q_network(torch.cat((state_batch, action_batch), dim=-1))
        Q1_loss = F.mse_loss(Q1_st_at, Q_hat_st_at)
        Q2_loss = F.mse_loss(Q2_st_at, Q_hat_st_at)
        Q_loss = Q1_loss + Q2_loss

        self.q_optim.zero_grad()
        Q_loss.backward()
        self.q_optim.step()

        # Update Policy
        actions, log_prob, _ = self.policy.sample(state_batch)
        Q1_st_at, Q2_st_at = self.q_network(torch.cat((state_batch, actions), dim=-1))
        Q_st_at = torch.min(Q1_st_at, Q2_st_at)
        policy_loss = ((self.alpha * log_prob) - Q_st_at).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update Value Average
        for target_param, param in zip(self.value_avg_network.parameters(), self.value_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        return value_loss.item(), Q1_loss.item(), Q2_loss.item(), policy_loss.item()
