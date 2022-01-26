import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

class ValueNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Policy(nn.Module):
    def __init__(self, input_size, output_size=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc_mean = nn.Linear(256, output_size)
        self.fc_log_std = nn.Linear(256, output_size)
        # don't do this
        # make it as an output of NN
        # self.log_std = nn.Parameter(torch.log(torch.ones(output_size, requires_grad=True)))
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        # Use std to be a layer output
        # It is not common to output a distribution
        return mean, log_std

    def sample(self, state):
        # batch_size * 1
        mean, log_std = self.forward(state)
        std = log_std.exp()
        distribution = Normal(mean, std)
        sample = distribution.rsample()
        # print(sample)
        actions = torch.tanh(sample)
        log_prob = distribution.log_prob(sample)
        log_prob -= torch.log(1 - torch.tanh(sample).pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)
        return actions, log_prob

# if __name__ == "__main__":
#     policy = Policy(2)
#     policy2 = Policy(2)
#     policy2.load_state_dict(policy.state_dict())
#     a = torch.tensor([1., 2.])
#     print(policy(a), policy2(a))
#     with torch.no_grad():
#         for p in policy.parameters():
#             p += torch.tensor(1.)
#     a = torch.tensor([1., 2.])
#     print(policy(a), policy2(a))
        


