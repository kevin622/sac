import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class ValueNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.apply(weights_init_)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_state, num_action, hidden_dim):
        super(QNetwork, self).__init__()
        # Q1
        self.fc1 = nn.Linear(num_state + num_action, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2
        self.fc4 = nn.Linear(num_state + num_action, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action):
        x_input = torch.cat([state, action], dim=1)
        x1 = F.relu(self.fc1(x_input))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x_input))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2


class Policy(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, action_space=None):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, output_size)
        self.fc_log_std = nn.Linear(hidden_dim, output_size)
        self.apply(weights_init_)

        if action_space == None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = ((action_space.high - action_space.low) / 2.)
            self.action_bias = ((action_space.high + action_space.low) / 2.)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Policy, self).to(device)
