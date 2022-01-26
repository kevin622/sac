import argparse

import wandb
import torch
import gym
from stable_baselines3.common.env_util import make_vec_env

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
    parser.add_argument("--num_iter",
                        default=int(1e6),
                        type=int,
                        help="Number of the Training Iteration (default: 1000000)")
    parser.add_argument("--max_env_step",
                        default=1000,
                        type=int,
                        help="Max length of the Environment Step (default: 1000)")
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

    env = make_vec_env(args.env_name, n_envs=args.env_nums)
    eval_env = gym.make(id=args.env_name) # Env that will be used for evaluation of the process

    # Agent
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SAC(device=device, args=args, env=env)

    # Weights and Biases(logging)
    wandb.init(project="sac_2", entity="kevin622")
    wandb.config = {
        "env_name" : args.env_name,
        "env_nums" : args.env_nums,
        "buffer_size" : args.buffer_size,
        "lr" : args.lr,
        "gamma" : args.gamma,
        "num_iter" : args.num_iter,
        "max_env_step" : args.max_env_step,
        "num_grad_step" : args.num_grad_step,
        "batch_size" : args.batch_size,
        "tau" : args.tau,
    }

    # Replay Buffer
    replay_buffer = ReplayBuffer(size=args.buffer_size, device=device, env=env)
    replay_buffer.generate_data()

    # Training Loop
    env = make_vec_env(args.env_name, 1)
    state = env.reset()
    for ith_iter in range(1, args.num_iter + 1):
        action = agent.get_action(state=to_tensor(state))
        next_state, reward, done, _ = env.step(action)
        # How can I ignore done being True due to hitting the time horizon
        # in Vectorized Env?
        replay_buffer.append([state, action, reward, next_state, done])
        state = next_state

        for ith_grad_step in range(args.num_grad_step):
            value_loss, Q1_loss, Q2_loss, policy_loss = agent.update_parameters(memory=replay_buffer, batch_size=args.batch_size)
            wandb.log({
                'value_loss' : value_loss, 
                'Q1_loss' : Q1_loss, 
                'Q2_loss' : Q2_loss, 
                'policy_loss' : policy_loss
            })

        if ith_iter % 100 == 0:
            '''
            For every 100 step, evaluate the policy by rolling it out
            '''
            eval_state = eval_env.reset()
            sum_reward = 0
            for ith_env_step in range(args.max_env_step):
                eval_action = agent.policy.sample(eval_state)
                eval_next_state, eval_reward, eval_done, _ =  eval_env.step(eval_action)
                sum_reward += eval_reward
                if eval_done:
                    break
                eval_state = eval_next_state
            cnt_env_step = ith_env_step + 1
            wandb.log({
                "sum_reward": sum_reward,
                "avg_reward": sum_reward / cnt_env_step,
            })


if __name__ == "__main__":
    main()
