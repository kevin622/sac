import argparse
import itertools
import os

import wandb
import torch
# from stable_baselines3.common.env_util import make_vec_env
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np

from sac import SAC
from replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description="Soft Actor Critic Args")
    parser.add_argument("--env_name",
                        default="Hopper-v2",
                        type=str,
                        help="Gym environment (default: Hopper-v2)")
    parser.add_argument("--gamma",
                        default=0.99,
                        metavar='G',
                        type=float,
                        help="Discount Rate of Future Values (default: 0.99)")
    parser.add_argument(
        "--tau",
        default=0.005,
        metavar='G',
        help=
        "Target Value Smoothing Constant. Large tau can lead to instabilities while small tau can make training slower. (default: 0.005)"
    )
    parser.add_argument("--lr",
                        default=0.0003,
                        metavar='G',
                        type=float,
                        help="Learning Rate of the Models (default: 0.0003)")
    parser.add_argument("--alpha",
                        default=0.2,
                        metavar='G',
                        type=float,
                        help="Temperature parameter for entropy importance (default: 0.2)")
    parser.add_argument("--seed", metavar='N', default=123456, type=int, help="Random Seed (default: 123456)")
    parser.add_argument("--batch_size", metavar='N', default=256, help="Size of a Batch (default: 256)")
    parser.add_argument("--num_step",
                        default=1000001,
                        metavar='N', 
                        type=int,
                        help="Max num of step (default: 1,000,000)")
    parser.add_argument("--hidden_dim", metavar='N', default=256, type=int, help="Dimension of hidden layer")
    parser.add_argument("--num_grad_step",
                        default=1,
                        metavar='N', 
                        type=int,
                        help="Number of Gradient Steps for each Iteration (default: 1)")
    parser.add_argument("--start_step",
                        default=10000,
                        metavar='N', 
                        type=int,
                        help="Steps for random action (default: 10,000)")
    parser.add_argument("--buffer_size",
                        default=1000000,
                        metavar='N', 
                        type=int,
                        help="Size of Replay Buffer (default: 1,000,000)")
    parser.add_argument('--cuda', action="store_true", help='Whether use CUDA(default: False)')
    args = parser.parse_args()

    env = gym.make(id=args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Agent
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.shape[0]
    agent = SAC(args, state_shape, action_shape)

    # Weights and Biases(logging)
    wandb.init(project="sac_3", entity="kevin622")
    wandb.config = {
        "env_name": args.env_name,
        "buffer_size": args.buffer_size,
        "lr": args.lr,
        "gamma": args.gamma,
        "alpha": args.alpha,
        "start_step": args.start_step,
        "num_step": args.num_step,
        "num_grad_step": args.num_grad_step,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "tau": args.tau,
    }

    # Replay Buffer
    replay_buffer = ReplayBuffer(args=args)

    # Training Loop
    total_step = 0
    for ith_episode in itertools.count(1):
        if total_step > args.num_step:
            break

        episode_length = 0
        episode_reward = 0
        done = False
        state = env.reset()
        # one episode
        while not done:
            if total_step < args.start_step:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, evaluation=False)
            # Parameter Update
            if len(replay_buffer) > args.batch_size:
                for ith_grad_step in range(args.num_grad_step):
                    Q1_loss, Q2_loss, policy_loss = agent.update_parameters(
                        replay_buffer, args.batch_size)
                    wandb.log({
                        'Q1_loss': Q1_loss,
                        'Q2_loss': Q2_loss,
                        'policy_loss': policy_loss,
                    })
            next_state, reward, done, _ = env.step(action)
            if episode_length == env._max_episode_steps:
                mask = 1
            else:
                mask = float(not done)
            # Save data in buffer
            replay_buffer.append(state, action, reward, next_state, mask)

            total_step += 1
            episode_length += 1
            episode_reward += reward
            state = next_state

        print(
            f'Episode: {ith_episode}, Length: {episode_length}, Reward: {round(episode_reward, 2)}, Total Step: {total_step}'
        )
        wandb.log({
            'episode_length': episode_length,
            'episode_reward': episode_reward,
        })

        # For evaluation
        if ith_episode % 10 == 0:
            episode_lengths = []
            episode_rewards = []
            for _ in range(10):
                done = False
                episode_length = 0
                episode_reward = 0
                state = env.reset()
                while not done:
                    action = agent.get_action(state, evaluation=True)
                    next_state, reward, done, _ = env.step(action)
                    episode_length += 1
                    episode_reward += reward
                    state = next_state
                episode_lengths.append(episode_length)
                episode_rewards.append(episode_reward)
            avg_episode_length = sum(episode_lengths) / len(episode_lengths)
            avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
            print('--------------------------')
            print(
                f'Evaluation on 10 episodes(average) - Length: {avg_episode_length}, Reward: {round(avg_episode_reward, 2)}'
            )
            print('--------------------------')
            wandb.log({
                "avg_episode_reward": avg_episode_reward,
                "avg_episode_length": avg_episode_length,
            })
            replay_buffer.save()

        # Video Record
        if ith_episode in {50, 100, *range(500, 100001, 500)}:
            if not os.path.exists("video/"):
                os.makedirs('video/')
            video_recorder = None
            video_recorder = VideoRecorder(env=env,
                                           base_path=f'video/{args.env_name}_{ith_episode}',
                                           enabled=True)
            state = env.reset()
            done = False
            while not done:
                env.render(mode='rgb_array')
                video_recorder.capture_frame()
                action = agent.get_action(state, evaluation=True)
                next_state, reward, done, _ = env.step(action)
                state = next_state
            video_recorder.close()
            video_recorder.enabled = False

    env.close()


if __name__ == "__main__":
    main()