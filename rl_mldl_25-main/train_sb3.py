"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3.

   TASK 4 & 5:
   - Train an RL policy (PPO or SAC) on the CustomHopper-source-v0 environment.
   - Evaluate the policy on source (lower bound) and target (sim-to-real transfer).
   - Optionally train directly on target to get an upper bound baseline.
"""

import argparse
import gym
from env.custom_hopper import *  # registers CustomHopper environments
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def make_vec_env(env_id: str, n_envs: int = 1, seed: int = 0):
    """Create a stable-baselines3-compatible vectorized environment."""
    def _make_env():
        def _init():
            env = gym.make(env_id)
            env.seed(seed)
            return env
        return _init

    return DummyVecEnv([_make_env() for _ in range(n_envs)])


def train_ppo(train_env_id: str,
              total_timesteps: int = 200_000,
              device: str = "cpu",
              n_envs: int = 1):

    print(f"Creating training env: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        device=device,
        tensorboard_log="./ppo_hopper_tb/",
    )

    print(f"Starting PPO training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")

    return model


def train_sac(train_env_id: str,
              total_timesteps: int = 200_000,
              device: str = "cpu",
              n_envs: int = 1):

    print(f"Creating training env for SAC: {train_env_id}")
    vec_env = make_vec_env(train_env_id, n_envs=n_envs)

    print("Observation space:", vec_env.observation_space)
    print("Action space:", vec_env.action_space)

    model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    device=device,
    seed=42,        # ensure reproducibility
    tensorboard_log="./sac_hopper_tb/",
     )


    print(f"Starting SAC training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")

    return model


def evaluate_on_env(model, env_id: str, n_episodes: int = 50):
    """Evaluate a trained model on the given environment."""
    print(f"\nEvaluating policy on env: {env_id}")
    eval_env = gym.make(env_id)

    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        render=False,
        deterministic=True,
    )

    print(f"Eval on {env_id} over {n_episodes} episodes:")
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    return mean_reward, std_reward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo",
                        help="RL algorithm: ppo or sac")
    parser.add_argument("--train-env", type=str, default="CustomHopper-source-v0",
                        help="Environment ID to train on.")
    parser.add_argument("--test-env", type=str, default="CustomHopper-target-v0",
                        help="Environment ID to test on.")
    parser.add_argument("--total-timesteps", type=int, default=200_000,
                        help="Number of training timesteps.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu or cuda.")
    return parser.parse_args()


def main():
    args = parse_args()
    algo = args.algo.lower()

    # ---- 1. TRAIN ----
    if algo == "ppo":
        model = train_ppo(
            train_env_id=args.train_env,
            total_timesteps=args.total_timesteps,
            device=args.device,
            n_envs=1,
        )
    elif algo == "sac":
        model = train_sac(
            train_env_id=args.train_env,
            total_timesteps=args.total_timesteps,
            device=args.device,
            n_envs=1,
        )
    else:
        raise NotImplementedError("Supported algorithms: ppo, sac")

   # ---- 2. SAVE MODEL ----
    suffix = args.train_env.replace("CustomHopper-", "").replace("-v0", "")
    save_name = f"{algo}_hopper_{suffix}"
    print(f"\nSaving model to: {save_name}.zip")
    model.save(save_name)

   # ---- 3. Evaluate on train and test envs ----
    evaluate_on_env(model, args.train_env, n_episodes=10)
    evaluate_on_env(model, args.test_env, n_episodes=10)


if __name__ == "__main__":
    main()
