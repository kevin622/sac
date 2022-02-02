from typing import OrderedDict
import torch
from torch.optim import Adam
import torch.nn.functional  as F
from models import Policy, QNetwork, ValueNetwork
from data_generation.data_generation import ReplayBuffer
from utils import hard_update, soft_update, to_numpy, to_tensor

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
        # Policy(actor)
        self.policy = Policy(input_size=state_shape, output_size=action_shape).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        # Q values(critic)
        self.q_network = QNetwork(num_state=state_shape, num_action=action_shape).to(device)
        self.q_optim = Adam(self.q_network.parameters(), lr=args.lr)

        self.q_target_network = QNetwork(num_state=state_shape, num_action=action_shape).to(device)
        hard_update(self.q_target_network, self.q_network)
        # # Value
        # self.value_network = ValueNetwork(input_size=state_shape).to(device)
        # self.value_optim = Adam(self.value_network.parameters(), lr=args.lr)
        # # Value Exponential Average
        # self.value_avg_network = ValueNetwork(input_size=state_shape).to(device)
        # hard_update(self.value_avg_network, self.value_network)

    def get_action(self, state, evaluation=False):
        state = to_tensor(state).unsqueeze(0)
        if evaluation:
            action = self.policy.sample(state)[2]
        else:
            action = self.policy.sample(state)[0]
        return to_numpy(action)[0]
        
    
    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        '''
        returns (value_loss, Q1_loss, Q2_loss, policy_loss)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.random_sample(size=batch_size)
        state_batch = to_tensor(state_batch)
        action_batch = to_tensor(action_batch)
        reward_batch = to_tensor(reward_batch).unsqueeze(1)
        next_state_batch = to_tensor(next_state_batch)
        mask_batch = to_tensor(mask_batch).unsqueeze(1)

        # # Update Value
        # with torch.no_grad():
        #     curr_policy_action_batch, curr_policy_log_prob, _ = self.policy.sample(state_batch)
        #     curr_policy_Q1_st_at, curr_policy_Q2_st_at = self.q_network(state_batch, curr_policy_action_batch)
        #     curr_policy_Q_st_at = torch.min(curr_policy_Q1_st_at, curr_policy_Q2_st_at)
        # V_st = self.value_network(state_batch)
        # value_loss = F.mse_loss(V_st, curr_policy_Q_st_at - self.alpha * curr_policy_log_prob)

        # self.value_optim.zero_grad()
        # value_loss.backward()
        # self.value_optim.step()
        
        # # Update Q
        # # Use the exponenetial average parameters for Value network
        # with torch.no_grad():
        #     V_st_1 = self.value_avg_network(next_state_batch)
        #     Q_hat_st_at = reward_batch + mask_batch * self.gamma * V_st_1
        # Q1_st_at, Q2_st_at = self.q_network(state_batch, action_batch)
        # Q1_loss = F.mse_loss(Q1_st_at, Q_hat_st_at)
        # Q2_loss = F.mse_loss(Q2_st_at, Q_hat_st_at)
        # Q_loss = Q1_loss + Q2_loss

        # self.q_optim.zero_grad()
        # Q_loss.backward()
        # self.q_optim.step()

        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_state_batch)
            Q1_next_target, Q2_next_target = self.q_target_network(next_state_batch, next_action)
            Q_next_target = torch.min(Q1_next_target, Q2_next_target) - self.alpha * next_log_prob
            Q_hat_st_at = reward_batch + mask_batch * self.gamma * Q_next_target
        Q1_st_at, Q2_st_at = self.q_network(state_batch, action_batch)
        Q1_loss = F.mse_loss(Q1_st_at, Q_hat_st_at)
        Q2_loss = F.mse_loss(Q2_st_at, Q_hat_st_at)
        Q_loss = Q1_loss + Q2_loss

        self.q_optim.zero_grad()
        Q_loss.backward()
        self.q_optim.step()


        # Update Policy
        actions, log_prob, _ = self.policy.sample(state_batch)
        Q1_st_at, Q2_st_at = self.q_network(state_batch, actions)
        Q_st_at = torch.min(Q1_st_at, Q2_st_at)
        policy_loss = ((self.alpha * log_prob) - Q_st_at).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        soft_update(self.q_target_network, self.q_network, self.tau)

        # # Update Value Average
        # soft_update(self.value_avg_network, self.value_network, self.tau)

        # return value_loss.item(), Q1_loss.item(), Q2_loss.item(), policy_loss.item()
        return Q1_loss.item(), Q2_loss.item(), policy_loss.item()
