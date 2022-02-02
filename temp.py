from stable_baselines3.common.env_util import make_vec_env
import gym

# env = make_vec_env("Hopper-v2", n_envs=2)
env = gym.make("Hopper-v2")
print(dir(env))
print(env._max_episode_steps)