"""Runs a saved policy on fixed seeds and counts how each episode ended.

Usage (from the python folder):
        ./.venv/Scripts/python.exe check_policy.py runs/<run-folder>
"""
import sys
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_env import RocketLandingEnv


def main():
    run_dir = sys.argv[1]
    model = PPO.load(f"{run_dir}/final_model")

    # Load the saved normalization statistics. Only the observation statistics are used here,
    # so reward normalization is switched off.
    dummy = DummyVecEnv([lambda: RocketLandingEnv(difficulty=0.0)])
    norm = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", dummy)
    norm.training = False
    norm.norm_reward = False

    outcomes = Counter()
    for seed in range(50):
        env = RocketLandingEnv(difficulty=0.0)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])
        outcomes["Timeout" if truncated else info["status"]] += 1

    print(dict(outcomes))


if __name__ == "__main__":
    main()