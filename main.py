import argparse
import itertools

import wandb
import torch
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np

from sac import SAC
from replay_buffer import ReplayBuffer
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
    parser.add_argument(
        "--num_iter",  # can be removed
        default=int(1e5),
        type=int,
        help="Number of the Training Iteration (default: 100,000)")
    parser.add_argument("--start_step",
                        default=10000,
                        type=int,
                        help="Steps for random action (default: 10,000)")
    parser.add_argument("--num_step",
                        default=1000000,
                        type=int,
                        help="Max num of step (default: 1,000,000)")
    parser.add_argument(
        "--max_env_step",  # can be removed
        default=1000,
        type=int,
        help="Max length of the Environment Step (default: 1000)")
    parser.add_argument("--num_grad_step",
                        default=1,
                        type=int,
                        help="Number of Gradient Steps for each Iteration (default: 1)")
    parser.add_argument("--batch_size", default=256, help="Size of a Batch (default: 256)")
    parser.add_argument("--seed", default=123456, help="Random Seed (default: 123456)")
    parser.add_argument(
        "--tau",
        default=0.005,
        help=
        "Target Value Smoothing Constant. Large tau can lead to instabilities while small tau can make training slower. (default: 0.005)"
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # env = make_vec_env(args.env_name, n_envs=args.env_nums)
    # env.seed(args.seed)
    # eval_env = make_vec_env(env_id=args.env_name,
    #                         n_envs=1)  # Env that will be used for evaluation of the process
    # eval_env.seed(args.seed)
    env = gym.make(id=args.env_name)
    env.seed(args.seed)
    # Agent
    agent = SAC(device=device, args=args, env=env)

    # Weights and Biases(logging)
    # wandb.init(project="sac_2", entity="kevin622")
    # wandb.config = {
    #     "env_name": args.env_name,
    #     "env_nums": args.env_nums,
    #     "buffer_size": args.buffer_size,
    #     "lr": args.lr,
    #     "gamma": args.gamma,
    #     "num_iter": args.num_iter,
    #     "max_env_step": args.max_env_step,
    #     "num_grad_step": args.num_grad_step,
    #     "batch_size": args.batch_size,
    #     "tau": args.tau,
    # }

    # Replay Buffer
    replay_buffer = ReplayBuffer(device=device, args=args)
    # replay_buffer.generate_data()

    # Training Loop

    total_step = 0
    for ith_episode in itertools.count(1):
        done = False
        state = env.reset()
        episode_length = 0
        episode_reward = 0
        # one episode
        while not done:
            if total_step < args.start_step:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, evaluation=False)
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
            
            # Parameter Update
            if len(replay_buffer) > args.batch_size:
                for ith_grad_step in range(args.num_grad_step):
                    Q1_loss, Q2_loss, policy_loss = agent.update_parameters(replay_buffer, args.batch_size)
                    # value_loss, Q1_loss, Q2_loss, policy_loss = agent.update_parameters(replay_buffer, args.batch_size)
                    # wandb.log({
                    #     'value_loss' : value_loss,
                    #     'Q1_loss' : Q1_loss,
                    #     'Q2_loss' : Q2_loss,
                    #     'policy_loss' : policy_loss,
                    # })
        
        print(f'Episode: {ith_episode}, Length: {episode_length}, Reward: {round(episode_reward, 2)}, Total Step: {total_step}')
        # wandb.log({
        #     'episode_length': episode_length,
        #     'episode_reward': episode_reward,
        # })

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
            print(f'Evaluation on 10 episodes(average) - Length: {avg_episode_length}, Reward: {round(avg_episode_reward, 2)}')
            print('--------------------------')
            # wandb.log({
            #     "sum_reward": avg_episode_reward,
            #     "episode_length": avg_episode_length,
            # })


                    


        # all_done = [0] * env.num_envs
        # all_rewards = [0] * env.num_envs
        # states = env.reset()

        # while sum(all_done) < env.num_envs:
        #     if total_step < args.start_step:
        #         actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
        #     else:
        #         actions = agent.get_action(states)

        #     if len(replay_buffer) > args.batch_size:
        #         for ith_grad_step in range(args.num_grad_step):
        #             value_loss, Q1_loss, Q2_loss, policy_loss = agent.update_parameters(
        #                 replay_buffer, args.batch_size)
        #         wandb.log({
        #             'value_loss': value_loss,
        #             'Q1_loss': Q1_loss,
        #             'Q2_loss': Q2_loss,
        #             'policy_loss': policy_loss,
        #         })

        #     next_states, rewards, dones, _ = env.step(actions)
        #     idx = -1
        #     for state, action, reward, next_state, done in zip(states, actions, rewards,
        #                                                        next_states, dones):
        #         replay_buffer.append(state, action, reward, next_state, done)
        #         total_step += 1
        #         idx += 1
        #         if not all_done[idx]:
        #             all_rewards[idx] += reward
        #         if done:
        #             all_done[idx] = 1
        #     states = next_states

        # print(
        #     f'Iter {ith_iter} average({env.num_envs} envs) sum reward: {round(sum(all_rewards) / env.num_envs, 2)}'
        # )
        # if total_step > args.num_step:
        #     break

        # if ith_iter % 10 == 0:
        #     sum_reward_list = []
        #     cnt_env_step_list = []
        #     for _ in range(10):
        #         eval_state = eval_env.reset()
        #         sum_reward = 0
        #         cnt_env_step = 0
        #         eval_done = False
        #         while not eval_done:
        #             eval_action = agent.policy.sample(to_tensor(eval_state))[2]
        #             eval_next_state, eval_reward, eval_done, _ = eval_env.step(
        #                 to_numpy(eval_action))
        #             sum_reward += eval_reward[0]
        #             eval_done = eval_done[0]
        #             if eval_done:
        #                 break
        #             eval_state = eval_next_state
        #             cnt_env_step += 1
        #         sum_reward_list.append(sum_reward)
        #         cnt_env_step_list.append(cnt_env_step)
        #     avg_sum_reward = sum(sum_reward_list) / len(sum_reward_list)
        #     avg_cnt_env_step = sum(cnt_env_step_list) / len(cnt_env_step_list)
        #     wandb.log({
        #         "sum_reward": avg_sum_reward,
        #         "episode_length": avg_cnt_env_step,
        #     })
        #     print('---------------')
        #     print(
        #         f'Evaluation in iter {ith_iter} - Total Steps: {total_step}, Sum of Reward: {round(avg_sum_reward, 2)}, Length of Episode: {avg_cnt_env_step} (10 episode average)'
        #     )
        #     print('---------------')
        #     replay_buffer.save()

if __name__ == "__main__":
    main()
