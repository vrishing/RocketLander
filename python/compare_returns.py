"""Compares the total reward of three behaviors on the same starting states.

Usage (from the python folder):
    ./.venv/Scripts/python.exe compare_returns.py
"""
import rocketphys_py as rp

HOVER_ACTION = [-0.2152, 0.0, 0.0]  # throttle 0.3924, the hover value
SEEDS = range(1, 21)


def run_episode(seed, choose_action):
    """Plays one episode from the given seed, returns (total reward, outcome)."""
    cfg = rp.EnvConfig()
    cfg.difficulty = 0.0                 # upright, still, 15 to 40 m start range
    env = rp.LandingEnv(cfg)
    env.reset(seed)

    total = 0.0
    while True:
        result = env.step(choose_action(env))
        total += result.reward
        if result.terminated or result.truncated:
            outcome = result.status.name if result.terminated else "Timeout"
            return total, outcome


def summarize(name, choose_action):
    totals, outcomes = [], {}
    for seed in SEEDS:
        total, outcome = run_episode(seed, choose_action)
        totals.append(total)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    mean = sum(totals) / len(totals)
    print(f"{name:10s} mean return {mean:8.2f}   outcomes {outcomes}")


def main():
    pilot = rp.LandingBaseline()
    summarize("baseline", lambda env: pilot.act(env))
    summarize("hover", lambda env: HOVER_ACTION)


if __name__ == "__main__":
    main()