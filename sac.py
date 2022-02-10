import torch
import torch.nn.functional as F
from torch.optim import Adam
from replay_buffer import ReplayBuffer
from utils import hard_update, soft_update, to_numpy, to_tensor
from models import Policy, QNetwork


class SAC(object):

    def __init__(self, args, state_shape, action_shape, action_space=None):
        # hyperparameters
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.device = torch.device("cuda" if args.cuda else "cpu")

        # Neural Nets
        # Q values(critic)
        self.critic = QNetwork(state_shape, action_shape, args.hidden_dim).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(state_shape, action_shape, args.hidden_dim).to(self.device)
        # copy the NN parameters
        hard_update(self.critic_target, self.critic)

        # Policy(actor)
        self.policy = Policy(state_shape, action_shape, args.hidden_dim, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def get_action(self, state, evaluation=False):
        '''
        If evaluation is False, it returns a sampled action from the distribution.
        If evaluation is True, it returns a mean action(action as the mean value) from the distribution
        '''
        state = to_tensor(state).to(self.device).unsqueeze(0)
        if not evaluation:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return to_numpy(action)[0]

    def update_parameters(self, memory: ReplayBuffer, batch_size: int, update_cnt: int):
        '''
        returns (value_loss, Q1_loss, Q2_loss, policy_loss)
        '''
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(
            size=batch_size)

        state_batch = to_tensor(state_batch).to(self.device)
        action_batch = to_tensor(action_batch).to(self.device)
        reward_batch = to_tensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = to_tensor(next_state_batch).to(self.device)
        mask_batch = to_tensor(mask_batch).to(self.device).unsqueeze(1)

        # Update Critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch,
                                                                  next_state_action)
            min_qf_next_target = torch.min(qf1_next_target,
                                           qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        Q1_st_at, Q2_st_at = self.critic(state_batch, action_batch)
        Q1_loss = F.mse_loss(Q1_st_at, next_q_value)
        Q2_loss = F.mse_loss(Q2_st_at, next_q_value)
        Q_loss = Q1_loss + Q2_loss

        self.critic_optim.zero_grad()
        Q_loss.backward()
        self.critic_optim.step()

        # Update Policy
        pi, log_pi, _ = self.policy.sample(state_batch)
        Q1_st_at_pi, Q2_st_at_pi = self.critic(state_batch, pi)
        Q_st_at_pi = torch.min(Q1_st_at_pi, Q2_st_at_pi)

        policy_loss = ((self.alpha * log_pi) - Q_st_at_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if update_cnt % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        return Q1_loss.item(), Q2_loss.item(), policy_loss.item()
