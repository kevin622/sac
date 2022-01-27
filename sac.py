from typing import OrderedDict
import torch
from torch.optim import Adam
import torch.nn.functional  as F
from models.models import Policy, QNetwork, ValueNetwork
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

        # # Policy
        # self.policy = Policy(input_size=state_shape, output_size=action_shape).to(device)
        # self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        # # Q values
        # self.q1_network = QNetwork(input_size=(state_shape + action_shape)).to(device)
        # self.q1_optim = Adam(self.q1_network.parameters(), lr=args.lr)
        # self.q2_network = QNetwork(input_size=(state_shape + action_shape)).to(device)
        # self.q2_optim = Adam(self.q2_network.parameters(), lr=args.lr)
        # # Value
        # self.value_network = ValueNetwork(input_size=state_shape).to(device)
        # self.value_optim = Adam(self.value_network.parameters(), lr=args.lr)
        # # Value Exponential Average
        # self.value_avg_network = ValueNetwork(input_size=state_shape).to(device)
        # self.value_avg_network.load_state_dict(self.value_network.state_dict())

    def get_action(self, state):
        state = to_tensor(state).unsqueeze(0)
        action = self.policy.sample(state)[0]
        return to_numpy(action)[0]
        
    
    def update_parameters(self, memory: ReplayBuffer, batch_size: int):
        '''
        returns (value_loss, Q1_loss, Q2_loss, policy_loss)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = memory.random_sample(size=batch_size)

        state_batch = to_tensor(state_batch)
        action_batch = to_tensor(action_batch)
        reward_batch = to_tensor(reward_batch)
        next_state_batch = to_tensor(next_state_batch)
        done_batch = to_tensor(done_batch)

        # Update critic(Q)
        with torch.no_grad():
            next_action_batch, next_action_log_prob = self.policy.sample(next_state_batch)
            q1_st1_at1, q2_st1_at1 =  self.critic_target(next_state_batch, next_action_batch)
            q_st1_at1 = torch.min(q1_st1_at1, q2_st1_at1) - self.alpha * next_action_log_prob
            q_target = reward_batch + done_batch * self.gamma * q_st1_at1
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
        policy_loss = (log_pi_action_batch * self.alpha - q_st_at_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        
        return q1_loss.item(), q2_loss.item(), policy_loss.item()





        # # Update Value
        # V_st = self.value_network(state_batch)
        # curr_policy_action_batch, curr_policy_log_prob = self.policy.sample(state_batch)
        # curr_policy_Q1_st_at = self.q1_network(torch.cat((state_batch, curr_policy_action_batch), dim=-1))
        # curr_policy_Q2_st_at = self.q2_network(torch.cat((state_batch, curr_policy_action_batch), dim=-1))
        # curr_policy_Q_st_at = torch.min(curr_policy_Q1_st_at, curr_policy_Q2_st_at)
        # value_loss = F.mse_loss(V_st, curr_policy_Q_st_at - curr_policy_log_prob)

        # self.value_optim.zero_grad()
        # value_loss.backward()
        # self.value_optim.step()
        
        # # Update Q
        # Q1_st_at = self.q1_network(torch.cat((state_batch, action_batch), dim=-1))
        # Q2_st_at = self.q2_network(torch.cat((state_batch, action_batch), dim=-1))
        # # Use the exponenetial average parameters for Value network
        # V_st_1 = self.value_avg_network(next_state_batch)
        # Q_hat_st_at = reward_batch + done_batch * self.gamma * V_st_1
        # Q1_loss = F.mse_loss(Q1_st_at, Q_hat_st_at)
        # Q2_loss = F.mse_loss(Q2_st_at, Q_hat_st_at)

        # self.q1_optim.zero_grad()
        # Q1_loss.backward(retain_graph=True)
        # self.q1_optim.step()
        # self.q2_optim.zero_grad()
        # Q2_loss.backward()
        # self.q2_optim.step()

        # Update Policy
        # actions, log_prob = self.policy.sample(state_batch)
        # Q1_st_at = self.q1_network(torch.cat((state_batch, actions), dim=-1))
        # Q2_st_at = self.q2_network(torch.cat((state_batch, actions), dim=-1))
        # Q_st_at = torch.min(Q1_st_at, Q2_st_at)
        # policy_loss = torch.mean(log_prob - Q_st_at)

        # self.policy_optim.zero_grad()
        # policy_loss.backward()
        # self.policy_optim.step()

        # Update Value Average
        # value_parameters = self.value_network.state_dict()
        # value_avg_parameters = self.value_avg_network.state_dict()
        # new_state_dict = OrderedDict()
        # for param, _ in self.value_network.named_parameters():
        #     new_state_dict[param] = value_parameters[param] * self.tau + value_avg_parameters[param] * (1 - self.tau)
        # self.value_avg_network.load_state_dict(new_state_dict)

        # return value_loss.item(), Q1_loss.item(), Q2_loss.item(), policy_loss.item()
