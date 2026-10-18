"""Runs a saved policy on fixed seeds and reports how each episode ended.

Usage (from the python folder):
    ./.venv/Scripts/python.exe check_policy.py runs/<run-folder>
"""
import sys
from collections import Counter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_env import RocketLandingEnv

SOFT_SPEED = 2.0   # m/s, must match kSoftLandingSpeed in Landing.h
SOFT_TILT = 10.0   # degrees, must match kSoftLandingTiltDeg in Landing.h


def main():
    run_dir = sys.argv[1]
    model = PPO.load(f"{run_dir}/final_model")

    # Load the saved normalization statistics. Reward normalization is not needed for evaluation.
    dummy = DummyVecEnv([lambda: RocketLandingEnv(difficulty=0.0)])
    norm = VecNormalize.load(f"{run_dir}/vecnormalize.pkl", dummy)
    norm.training = False
    norm.norm_reward = False

    outcomes = Counter()
    crash_causes = Counter()
    landings = []   # (speed, tilt) for every episode that touched down

    for seed in range(50):
        env = RocketLandingEnv(difficulty=0.0)
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, _ = model.predict(norm.normalize_obs(obs[None, :]), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action[0])

        if truncated:
            outcomes["Timeout"] += 1
            continue

        status = info["status"]
        outcomes[status] += 1
        speed = info["touchdown_speed"]
        tilt = info["touchdown_tilt_deg"]
        landings.append((speed, tilt))

        if status == "Crashed":
            too_fast = speed >= SOFT_SPEED
            too_tilted = tilt >= SOFT_TILT
            if too_fast and too_tilted:
                crash_causes["too fast and too tilted"] += 1
            elif too_fast:
                crash_causes["too fast"] += 1
            else:
                crash_causes["too tilted"] += 1

    print("Outcomes:", dict(outcomes))
    print("Crash causes:", dict(crash_causes))
    print("Touchdowns (speed m/s, tilt deg):")
    for speed, tilt in sorted(landings):
        print(f"  {speed:6.2f}   {tilt:6.2f}")


if __name__ == "__main__":
    main()