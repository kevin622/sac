import torch
from torch.optim import Adam
import torch.nn.functional  as F
from models_2 import Policy, QNetwork
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
        # Critic - Q values
        self.critic = QNetwork(num_state=state_shape, num_action=action_shape).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(num_state=state_shape, num_action=action_shape).to(device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        # Actor - Policy
        self.policy = Policy(input_size=state_shape, output_size=action_shape).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)


    def get_action(self, state):
        state = to_tensor(state).unsqueeze(0)
        action = self.policy.sample(state)[0]
        return to_numpy(action)[0]
        
    
    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        '''
        returns (Q1_loss, Q2_loss, policy_loss)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.random_sample(size=batch_size)

        state_batch = to_tensor(state_batch)
        action_batch = to_tensor(action_batch)
        reward_batch = to_tensor(reward_batch).unsqueeze(1)
        next_state_batch = to_tensor(next_state_batch)
        mask_batch = to_tensor(mask_batch).unsqueeze(1)

        # Update critic(Q)
        with torch.no_grad():
            next_action_batch, next_action_log_prob = self.policy.sample(next_state_batch)
            q1_st1_at1, q2_st1_at1 =  self.critic_target(next_state_batch, next_action_batch)
            q_st1_at1 = torch.min(q1_st1_at1, q2_st1_at1) - self.alpha * next_action_log_prob
            q_target = reward_batch + mask_batch * self.gamma * q_st1_at1
        q1_st_at, q2_st_at = self.critic(state_batch, action_batch)
        q1_loss = F.mse_loss(q1_st_at, q_target)
        q2_loss = F.mse_loss(q2_st_at, q_target)
        q_loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        # Update Policy
        pi_action_batch, log_pi_action_batch = self.policy.sample(state_batch)
        q1_st_at_pi, q2_st_at_pi = self.critic(state_batch, pi_action_batch)
        q_st_at_pi = torch.min(q1_st_at_pi, q2_st_at_pi)
        policy_loss = (self.alpha * log_pi_action_batch - q_st_at_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item()
