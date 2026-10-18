"""Trains a PPO policy on the rocket landing environment.

Run from the python folder with the venv interpreter:
    ./.venv/Scripts/python.exe train.py --smoke   short check run, 50k steps
    ./.venv/Scripts/python.exe train.py           main run
"""
import argparse
import os
from collections import deque
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from rocket_env import RocketLandingEnv

N_ENVS = 6            # parallel copies of the environment, one per worker process
BASE_SEED = 1000      # worker i starts its random stream at BASE_SEED + i
SMOKE_STEPS = 50_000
MAIN_STEPS = 3_000_000


def make_env(rank: int, difficulty: float):
    """Returns a function that builds one wrapped environment inside a worker process."""
    def _init():
        env = Monitor(RocketLandingEnv(difficulty=difficulty))
        # Seeding the first reset fixes this worker's random stream. Later resets draw from it,
        # so every worker plays different episodes, and the whole run is reproducible.
        env.reset(seed=BASE_SEED + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Raises the start difficulty once the recent landing rate is high enough."""

    def __init__(self, start: float = 0.0, step: float = 0.1,
                 window: int = 300, threshold: float = 0.7, verbose: int = 1):
        super().__init__(verbose)
        self.difficulty = start
        self.step = step
        self.threshold = threshold
        self.outcomes = deque(maxlen=window)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            # Monitor adds the "episode" key on the step where an episode ends.
            # The same step's info carries the landing status set by RocketLandingEnv.
            if "episode" in info:
                self.outcomes.append(info.get("status") == "Landed")

        if len(self.outcomes) == self.outcomes.maxlen:
            rate = sum(self.outcomes) / len(self.outcomes)
            self.logger.record("curriculum/landing_rate", rate)
            if rate >= self.threshold and self.difficulty < 1.0:
                self.difficulty = min(1.0, self.difficulty + self.step)
                self.training_env.env_method("set_difficulty", self.difficulty)
                self.outcomes.clear()
                if self.verbose:
                    print(f"Curriculum: difficulty raised to {self.difficulty:.1f}")

        self.logger.record("curriculum/difficulty", self.difficulty)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="short run, no curriculum")
    parser.add_argument("--steps", type=int, default=MAIN_STEPS, help="total training steps")
    parser.add_argument("--name", default="ppo", help="run name, used in the folder name")
    args = parser.parse_args()

    total_steps = SMOKE_STEPS if args.smoke else args.steps
    run_dir = os.path.join("runs", f"{datetime.now():%Y%m%d-%H%M%S}-{args.name}")
    os.makedirs(run_dir, exist_ok=True)

    # Start upright and still. The curriculum raises difficulty during the main run.
    start_difficulty = 0.0
    venv = SubprocVecEnv([make_env(rank, start_difficulty) for rank in range(N_ENVS)])

    # VecNormalize keeps running averages of observations and rewards, so the network sees
    # numbers near zero with unit spread. Its statistics must be saved with the model.
    # clip_reward is large because the terminal rewards (+100 and -100) are the main signal.
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=100.0, gamma=0.999)

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=2048,               # steps per environment before each update
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.999,                # long horizon: a landing can take 20 s of simulated time
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=run_dir,
        device="cpu",               # a small MLP trains faster on CPU than on the GPU
        seed=0,
        verbose=1,
    )

    callbacks = [
        # Saves the model and the normalization statistics together every ~100k steps.
        CheckpointCallback(
            save_freq=max(1, 100_000 // N_ENVS),   # counted in vector steps, N_ENVS per step
            save_path=os.path.join(run_dir, "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ),
    ]
    if not args.smoke:
        callbacks.append(CurriculumCallback(start=start_difficulty))

    model.learn(total_timesteps=total_steps, callback=callbacks, tb_log_name="ppo")

    model.save(os.path.join(run_dir, "final_model"))
    venv.save(os.path.join(run_dir, "vecnormalize.pkl"))
    venv.close()
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()