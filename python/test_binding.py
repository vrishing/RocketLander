"""Smoke test and speed benchmark for the rocketphys_py module.

Run this from the python/ folder, using the same Python the module was built against.
"""
import random
import time

import rocketphys_py as rp


def check_import_and_reset():
    env = rp.LandingEnv(rp.EnvConfig())
    obs = env.reset(1)
    assert len(obs) == 18, f"expected 18 observation values, got {len(obs)}"
    assert env.observation_size == 18
    print("import ok, observation size:", len(obs))


def check_episode_ends():
    env = rp.LandingEnv(rp.EnvConfig())
    env.reset(7)
    rng = random.Random(7)
    for step in range(5000):
        action = [rng.uniform(-1.0, 1.0) for _ in range(3)]
        result = env.step(action)
        if result.terminated or result.truncated:
            print(f"episode ended after {step + 1} policy steps, "
                  f"status={result.status}, reward={result.reward:.2f}")
            return
    raise AssertionError("episode did not end within 5000 policy steps")


def benchmark(policy_steps=200_000):
    cfg = rp.EnvConfig()
    env = rp.LandingEnv(cfg)
    rng = random.Random(0)
    actions = [[rng.uniform(-1.0, 1.0) for _ in range(3)] for _ in range(1024)]

    env.reset(1)
    start = time.perf_counter()
    for i in range(policy_steps):
        result = env.step(actions[i % len(actions)])
        if result.terminated or result.truncated:
            env.reset(i + 2)
    elapsed = time.perf_counter() - start

    policy_rate = policy_steps / elapsed
    physics_rate = policy_rate * cfg.action_repeat
    print(f"{policy_rate:,.0f} policy steps/s, {physics_rate:,.0f} physics steps/s")


if __name__ == "__main__":
    check_import_and_reset()
    check_episode_ends()
    benchmark()