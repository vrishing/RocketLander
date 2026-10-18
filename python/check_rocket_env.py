"""Runs Gymnasium's official checker, plus a few quick checks of the wrapper."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from rocket_env import RocketLandingEnv


def main():
    # The official checker: validates spaces, reset, step, seeding and return types.
    check_env(RocketLandingEnv(), skip_render_check=True)
    print("check_env passed (default config)")

    # The wind sensor changes the observation size, so check that too.
    env = RocketLandingEnv(wind_sensor=True)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (21,), obs.shape
    print("wind sensor observation size: 21 ok")

    # Same seed must give the same start.
    a, _ = RocketLandingEnv().reset(seed=5)
    b, _ = RocketLandingEnv().reset(seed=5)
    assert np.array_equal(a, b)
    print("seeded reset is reproducible")

    # Difficulty can be changed on a live environment.
    env = RocketLandingEnv(difficulty=0.0)
    env.set_difficulty(1.0)
    env.reset(seed=2)
    print("set_difficulty ok")


if __name__ == "__main__":
    main()