"""Train an RL agent on the OpenAI Gym Hopper environment using
    REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym
from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--baseline', default=None, type=float,
                        help='Constant baseline value for REINFORCE (None = no baseline)')
    parser.add_argument('--algo', default='reinforce', type=str,
                        help='Algorithm: reinforce or actor_critic')
    parser.add_argument('--env', default='CustomHopper-source-v0', type=str, help='Environment name')
    return parser.parse_args()


args = parse_args()


def main():

    env = gym.make(args.env)
    
    # Print environment info
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    env.reset()  # Ensure domain-specific parameters (e.g., masses) are set before printing
    print('Dynamics parameters:', env.get_parameters())

    """
        Training
    """
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    # Initialize policy and agent
    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy,
                  device=args.device,
                  baseline_value=args.baseline,
                  algo=args.algo)

    #
    # TASK 2 and 3: interleave data collection to policy updates
    #
    for episode in range(args.n_episodes):
        done = False
        train_reward = 0
        state = env.reset()

        while not done:

            action, action_log_prob = agent.get_action(state)
            previous_state = state

            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_log_prob, reward, done)

            train_reward += reward

        # Update policy after each episode
        loss = agent.update_policy()

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode + 1)
            print('Episode return:', train_reward)
            if loss is not None:
                print('Loss:', loss)
    # Save final trained policy
    torch.save(agent.policy.state_dict(), "model.mdl")


if __name__ == '__main__':
    main()
