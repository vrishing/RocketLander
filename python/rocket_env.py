"""Gymnasium wrapper around the C++ rocket landing environment.

Training code uses this class. It converts between the C++ types from rocketphys_py
and the NumPy arrays and tuples that Gymnasium and stable-baselines3 expect.
"""
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import rocketphys_py as rp

ACTION_LIMIT = 1.0   # the C++ side clamps every action to [-1, 1]
OBS_LIMIT = 10.0     # the C++ side clamps every observation to [-10, 10]


class RocketLandingEnv(gym.Env):
    """A rocket that must land softly on the ground.

    Action  : float32 array of 3 values in [-1, 1]: throttle, pitch, roll.
    Observation: float32 array, 18 values, or 21 with wind_sensor=True.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        difficulty: float = 0.0,
        wind_sensor: bool = False,
        atmosphere: bool = False,
        max_wind_speed: float = 0.0,
        action_repeat: int = 4,
        max_episode_seconds: float = 60.0,
    ):
        super().__init__()

        # Build the C++ config once. Atmosphere, wind and the wind sensor are fixed here,
        # because they change the physics setup or the observation size.
        cfg = rp.EnvConfig()
        cfg.difficulty = float(difficulty)
        cfg.wind_sensor = bool(wind_sensor)
        cfg.atmosphere = bool(atmosphere)
        cfg.max_wind_speed = float(max_wind_speed)
        cfg.action_repeat = int(action_repeat)
        cfg.max_episode_seconds = float(max_episode_seconds)
        self._env = rp.LandingEnv(cfg)

        n_obs = self._env.observation_size
        self.action_space = gym.spaces.Box(
            low=-ACTION_LIMIT, high=ACTION_LIMIT, shape=(3,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-OBS_LIMIT, high=OBS_LIMIT, shape=(n_obs,), dtype=np.float32
        )

    # ------------------------------------------------------------------ Gymnasium API

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Gymnasium's base class sets self.np_random from the seed. Use it to choose
        # the C++ seed when none was given, so the wrapper stays reproducible.
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**32 - 1))
        obs = self._env.reset(int(seed) & 0xFFFFFFFF)
        return self._to_array(obs), {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Clip here as well, so a slightly out-of-range policy output never reaches C++.
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -ACTION_LIMIT, ACTION_LIMIT)
        result = self._env.step([float(x) for x in a])

        info = {
            "status": result.status.name,       # "Flying", "Landed" or "Crashed"
            "height": self._env.height,         # altitude of the lowest point, m
            "tilt_deg": self._env.tilt_deg,     # angle from upright, degrees
        }
        return (
            self._to_array(result.observation),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            info,
        )

    # ------------------------------------------------------------------ Curriculum

    def set_difficulty(self, difficulty: float) -> None:
        """Change the start range during training. 0 = upright and still, 1 = fully random."""
        self._env.set_difficulty(float(difficulty))

    # ------------------------------------------------------------------ Helpers

    @staticmethod
    def _to_array(values) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)