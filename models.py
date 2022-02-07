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
    def __init__(self, num_state, num_action):
        super(QNetwork, self).__init__()
        # Q1
        self.fc1 = nn.Linear(num_state + num_action, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
        # Q2
        self.fc4 = nn.Linear(num_state + num_action, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

        self.apply(weights_init_)
    
    def forward(self, state, action):
        x_input = torch.cat((state, action), dim=1)
        x1 = F.relu(self.fc1(x_input))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x_input))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2


class Policy(nn.Module):
    def __init__(self, input_size, output_size, action_space=None):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc_mean = nn.Linear(256, output_size)
        self.fc_log_std = nn.Linear(256, output_size)
        self.apply(weights_init_)

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
        sample = normal.rsample()
        action = torch.tanh(sample)

        log_prob = normal.log_prob(sample)
        log_prob -= torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action
