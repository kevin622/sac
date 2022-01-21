import random
from collections import OrderedDict

import numpy as np
import torch
from torch.optim import Adam
import gym
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
from matplotlib import pyplot as plt
import wandb
wandb.init(project="sac", entity="kevin622")

from models.models import Value, Q_value, Policy
from data_generation.data_generation import ReplayBuffer

# GPU support if cuda is available
'''
CUDA_VISIBLE_DEVICES=1,2 python train.py
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# parallel environments
n_envs = 4
env = make_vec_env("Hopper-v2", n_envs=n_envs)

# initial policy
state_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
policy = Policy(input_size=state_shape, output_size=action_shape).to(device)


# initial dataset
buffer_size = int(1e6)

replay_buffer = ReplayBuffer(size=buffer_size, device=device, env=env)
replay_buffer.generate_data(policy=policy)


# initial value function and q function
value = Value(input_size=state_shape).to(device)
value_bar = Value(input_size=state_shape).to(device)
value_bar.load_state_dict(value.state_dict())
q_value_1 = Q_value(input_size=(state_shape + action_shape)).to(device)
q_value_2 = Q_value(input_size=(state_shape + action_shape)).to(device)

# Backprop
lr = 3 * 1e-4
discount = 0.99

## Value
optimizer_value = Adam(value.parameters(), lr=lr)
# def loss_value(s_t, a_t):
def loss_value(v, q, log_pi):
    return -torch.mean(torch.square(v - q + log_pi))

## Q
optimizer_q1 = Adam(q_value_1.parameters(), lr=lr)
optimizer_q2 = Adam(q_value_2.parameters(), lr=lr)
def loss_q1(q1, r_t, v_bar, discount=discount):
    return -torch.mean(torch.square(q1 - r_t - (discount * v_bar)))

def loss_q2(q2, r_t, v_bar, discount=discount):
    return -torch.mean(torch.square(q2 - r_t - (discount * v_bar)))

## policy
optimizer_policy = Adam(policy.parameters(), lr=lr)
# def loss_policy(s_t):
def loss_policy(log_pi, q):
    return -torch.mean(log_pi - q)
    # return -torch.mean(log_pi * (log_pi - q))

# The Algorithm
whole_iter = int(1e6)
max_env_step = 1000  # max length of env step
whole_grad = 1
batch_size = 256
tau = 0.005  # target smoothing coefficient
## Make sure it runs in at least one env
## reproduce results of paper

print('Start Learning')
eval_env = gym.make("Hopper-v2") # env for evaluating
one_env = gym.make("Hopper-v2") # env for one step saving
one_obs = one_env.reset()
average_rewards = []
sum_rewards = []

# checking all the parameters
wandb.config = {
    "n_envs": n_envs,
    "buffer_size": buffer_size,
    "learning_rate": lr,
    "discount": discount,
    "whole_iter": whole_iter,
    "max_env_step" : max_env_step,
    "whole_grad" : whole_grad,
    "batch_size" : batch_size,
    "tau" : tau,
}


for each_iter in tqdm(range(whole_iter)):
    # For evaluating the our fit every 100 step
    
    if each_iter % 100 == 0:
        observation = eval_env.reset()
        reward_sum = 0
        for each_env in range(max_env_step):
            action = torch.tanh(policy(torch.from_numpy(observation).to(torch.float32).to(device)).sample()).to("cpu").numpy()
            next_observation, reward, done, info = eval_env.step(action)
            replay_buffer.append([observation, action, reward, next_observation])
            observation = next_observation
            reward_sum += reward
            if done:
                average_rewards.append(reward_sum / (each_env + 1))
                sum_rewards.append(reward_sum)
                break
        wandb.log({
            'reward_sum': sum_rewards[-1],
            'average_reward': average_rewards[-1]
        })
    if each_iter % 10000 == 0:
        print('reward_sum;', sum_rewards[-1])
        print('average_reward;', average_rewards[-1])
    
    # wandb.watch(policy)
    # print('error value', torch.tanh(policy(torch.from_numpy(one_obs).to(torch.float32).to(device)).sample()))
    one_act = torch.tanh(policy(torch.from_numpy(one_obs).to(torch.float32).to(device)).sample()).to("cpu").numpy()
    next_one_obs, one_re, one_done, one_info = one_env.step(one_act)
    replay_buffer.append([one_obs, one_act, one_re, next_one_obs])
    if one_done:
        one_obs = one_env.reset()
    else:
        one_obs = next_one_obs

    
    for each_grad in range(whole_grad):
        s_t, a_t, r_t, s_t_1 = replay_buffer.random_sample(size=batch_size)
        pi = policy(s_t)
        
        # Update value
        v = value(s_t)
        # new action from current policy
        u_t_curr_policy = pi.sample()
        # apply tanh
        a_t_curr_policy = torch.tanh(u_t_curr_policy)
        log_pi_curr_policy = pi.log_prob(u_t_curr_policy) - torch.sum(torch.log(1 - torch.square(torch.tanh(u_t_curr_policy))), dim=0)
        q1_curr_policy = q_value_1(torch.cat((s_t, a_t_curr_policy), dim=-1))
        q2_curr_policy = q_value_2(torch.cat((s_t, a_t_curr_policy), dim=-1))
        q_curr_policy = torch.min(q1_curr_policy, q2_curr_policy)

        loss = loss_value(v, q_curr_policy, log_pi_curr_policy)
        optimizer_value.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_value.step()

        # Update Q1
        v_bar = value_bar(s_t_1)
        q1 = q_value_1(torch.cat((s_t, a_t), dim=-1))

        loss = loss_q1(q1, r_t, v_bar, discount=discount)
        optimizer_q1.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_q1.step()

        # Update Q2
        q2 = q_value_2(torch.cat((s_t, a_t), dim=-1))
        loss = loss_q2(q2, r_t, v_bar, discount=discount)
        optimizer_q2.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_q2.step()
        
        # Update Policy
        # Evaluate the q again, since updated
        q1_curr_policy = q_value_1(torch.cat((s_t, a_t_curr_policy), dim=-1))
        q2_curr_policy = q_value_2(torch.cat((s_t, a_t_curr_policy), dim=-1))
        q_curr_policy = torch.min(q1_curr_policy, q2_curr_policy)

        loss = loss_policy(log_pi_curr_policy, q_curr_policy)
        optimizer_policy.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_policy.step()

        # Update Value_bar
        new_state_dict = OrderedDict()
        value_parameters = value.state_dict()
        value_bar_parameters = value_bar.state_dict()
        for param_name, _ in value.named_parameters():
            new_state_dict[param_name] = value_parameters[param_name] * tau + value_bar_parameters[param_name] * (1 - tau)
        value_bar.load_state_dict(new_state_dict)

print('done learning!')

print('saving the rewards')
with open("./figures/average_rewards.txt", "w") as file:
    for ar in average_rewards:
        file.writelines(str(ar))
        file.writelines('\n')

with open("./figures/sum_rewards.txt", "w") as file:
    for sr in sum_rewards:
        file.writelines(str(sr))
        file.writelines('\n')

# Also making and saving a plot
plt.figure()
plt.plot(average_rewards)
plt.title("Change of average of rewards")
plt.xlabel("Iteration")
plt.ylabel("average rewards")
plt.savefig("figures/average_rewards_plot.png")

plt.figure()
plt.plot(sum_rewards)
plt.title("Change of sum of rewards")
plt.xlabel("Iteration")
plt.ylabel("sum rewards")
plt.savefig("figures/sum_rewards_plot.png")
print('saved!')