import argparse
import itertools

import wandb
import torch
import gym
from stable_baselines3.common.env_util import make_vec_env

from sac_2 import SAC
from replay_buffer_2 import ReplayBuffer
from utils import to_numpy, to_tensor, device


def main():
    parser = argparse.ArgumentParser(description="Soft Actor Critic Args")
    parser.add_argument("--env_name",
                        default="Hopper-v2",
                        type=str,
                        help="Gym environment (default: Hopper-v2)")
    parser.add_argument("--env_nums",
                        default=4,
                        type=int,
                        help="Number of parallel Gym environments (default: 4)")
    parser.add_argument("--buffer_size",
                        default=int(1e6),
                        type=int,
                        help="Size of Replay Buffer (default: 1000000)")
    parser.add_argument("--lr",
                        default=3 * 1e-4,
                        type=float,
                        help="Learning Rate of the Models (default: 0.0003)")
    parser.add_argument("--gamma",
                        default=0.99,
                        type=float,
                        help="Discount Rate of Future Values (default: 0.99)")
    parser.add_argument("--alpha",
                        default=0.2,
                        type=float,
                        help="Temperature parameter for entropy importance (default: 0.2)")
    parser.add_argument("--num_step",
                        default=int(1e6),
                        type=int,
                        help="Number of the steps for whole training (default: 1,000,000)")
    parser.add_argument("--start_step",
                        default=int(1e4),
                        type=int,
                        help="Number of the random steps (default: 10,000)")
    parser.add_argument("--max_env_step",
                        default=1000,
                        type=int,
                        help="Max length of the Environment Step (default: 1,000)")
    parser.add_argument("--num_grad_step",
                        default=1,
                        type=int,
                        help="Number of Gradient Steps for each Iteration (default: 1)")
    parser.add_argument("--batch_size", default=256, help="Size of a Batch (default: 256)")
    parser.add_argument(
        "--tau",
        default=0.005,
        help=
        "Target Value Smoothing Constant. Large tau can lead to instabilities while small tau can make training slower. (default: 0.005)"
    )
    args = parser.parse_args()
    env = gym.make(id=args.env_name)

    # Agent
    agent = SAC(device=device, args=args, env=env)

    # Weights and Biases(logging)
    wandb.init(project="sac_2", entity="kevin622")
    wandb.config = {
        "env_name" : args.env_name,
        "env_nums" : args.env_nums,
        "buffer_size" : args.buffer_size,
        "lr" : args.lr,
        "gamma" : args.gamma,
        "num_step" : args.num_step,
        "start_step": args.start_step,
        "max_env_step" : args.max_env_step,
        "num_grad_step" : args.num_grad_step,
        "batch_size" : args.batch_size,
        "tau" : args.tau,
        "alpha": args.alpha,
    }

    # Replay Buffer
    replay_buffer = ReplayBuffer(size=args.buffer_size, device=device)

    # Training Loop
    num_total_step = 0
    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_length = 0
        done = False
        state = env.reset()
        while not done:
            if num_total_step < args.start_step: # Random Sample Action
                action = env.action_space.sample()
            else:
                action = agent.get_action(state)
            if len(replay_buffer) > args.batch_size:
                # Update the parameters
                q1_loss, q2_loss, policy_loss = agent.update_parameters(memory=replay_buffer, batch_size=args.batch_size)
                wandb.log({
                    "q1_loss": q1_loss,
                    "q2_loss": q2_loss,
                    "policy_loss": policy_loss,
                })

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            num_total_step += 1
            mask = 1 if episode_length == env._max_episode_steps else float(not done)
            replay_buffer.append(state, action, reward, next_state, mask)
            state = next_state
        print(f'Episode: {i_episode}, Length: {episode_length}, Reward: {round(episode_reward, 2)}, Total Steps: {num_total_step}')
        wandb.log({
            "Episode Sum of Reward": episode_reward,
            "Episode Average of Reward": episode_reward / episode_length,
        })
        if num_total_step > args.num_step:
            break


if __name__ == "__main__":
    main()
