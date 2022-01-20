import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

env = gym.make("Hopper-v2")
obs = env.reset()
rewards = []
cnt = 0
summed = 0
for i in tqdm(range(10000)):
    action = env.action_space.sample()
    obs, reward, done, info  = env.step(action)
    cnt += 1
    summed += reward
    if done:
        rewards.append(summed)
        cnt = 0
        summed = 0
        env.reset()


# with open("temp.txt", "r") as file:
#     nums = list(map(lambda x: float(x.rstrip()), file.readlines())) 
# print(nums)

plot = plt.plot(rewards)
plt.savefig("temp_plot.png")