# Rocket Lander

A headless C++ rigid-body physics engine and reinforcement-learning environment for learning to land a thruster-controlled rocket from random heights and orientations.

> **Status:** The physics and RL environment are complete and tested. A hand-designed baseline lands reliably from upright starts. PPO policies trained so far learn to hover until the time limit instead of landing. Reward redesign and curriculum tuning are in progress.

![Hero GIF](docs/hero.gif)

---

## What's built

### Physics library (`physics/`, C++17, no graphics dependency)
- Rigid body with unit-quaternion orientation, semi-implicit Euler integration, and world-space inverse inertia (R · I⁻¹ · Rᵀ) for cylinders and boxes.
- Forces applied off the center of mass produce torque (r × F).
- Nine-thruster rocket: a main engine plus four opposing RCS pairs for pitch and roll. Diagonal pairs give pure torque with no net force.
- Ground contact at the base rim, using impulses with restitution and Coulomb friction.
- Optional legs that latch out when the rocket descends below 15 m.
- Air drag in an exponential atmosphere, computed from velocity relative to moving air. Includes seeded gusts and a steady base wind.
- Drag acts at a center of pressure below the center of mass, which produces a weathervaning torque.
- Landing classification (Landed or Crashed) from touchdown speed and tilt.
- Fixed 1/120 s timestep, so runs are deterministic for a given seed.

### RL environment (`physics/LandingEnv`)
- Seeded reset with random height, orientation, and velocity. Difficulty scales the orientation and velocity ranges.
- Continuous action: throttle, pitch, and roll, each in [−1, 1].
- Observation: 18 values (21 with the optional wind sensor), clamped to [−10, 10].
- Potential-based altitude shaping plus per-step costs for tilt, spin, and speed outside a braking envelope.
- Terminal rewards for landing, crashing, and timing out.

### Baseline controller (`physics/Baseline`)
- Hand-designed landing controller. It uses the same action format as the policy, so both are scored in the same environment.

### Python (`python/`)
- `rocketphys_py`: pybind11 module exposing the environment, configuration, and baseline.
- `rocket_env.py`: Gymnasium wrapper.
- `train.py`: PPO with six parallel environments, VecNormalize, a difficulty curriculum, and TensorBoard logging.
- `check_policy.py`: evaluates a saved policy on fixed seeds and reports outcomes, crash causes, and touchdown speeds.
- `compare_returns.py`: compares total reward across baseline and hover behaviors.

### Viewer (`viewer/`)
- SDL2 and OpenGL window showing the rocket, thrust arrows, and leg lines.
- Manual thruster test mode, which fires each thruster group with the number keys.

---

## Results

### Physics and environment benchmark
| Measurement | Result | Conditions |
|---|---|---|
| Physics throughput | ~1.31 M steps/s per core | Release build, Windows, Python 3.13 bindings, random actions, one environment |
| Policy-step throughput | ~328 k steps/s | Same setup, 4 physics steps per policy step |

### Hand-designed baseline
| Start condition | Landed | Notes |
|---|---|---|
| Upright, still | 100 / 100 | Seeds 1–100 |
| Random orientation (first version) | 23 / 100 | Failed on starts past 90° |
| Random orientation (with flip recovery) | 62 / 100 | Seeds 1–100, full difficulty |

Failure analysis of the baseline, by starting tilt (flip version):

| Starting tilt | Landed |
|---|---|
| 0–45° | 16 / 18 |
| 45–90° | 23 / 31 |
| 90–135° | 20 / 40 |
| 135–180° | 3 / 11 |

The inverted starts are limited by physics. With RCS and a single main engine, an inverted rocket starting low can't recover before it reaches the ground. Some random starts are therefore unrecoverable, so the ceiling for full-random success is below 100%.

### Learned policies (PPO, deterministic evaluation, 50 seeds)
| Run | Reward design | Landed | Crashed | Timeout |
|---|---|---|---|---|
| Smoke test (50k steps) | Original | — | — | — |
| Main run | Original, γ = 0.995 | 0 | 0 | 50 |
| Shaped run | Altitude shaping, γ = 0.999, timeout penalty | 0 | 16 (all too fast) | 34 |
| Envelope run | Adds speed envelope | 0 | 0 | 50 |

No learned policy has landed reliably yet. Every failure mode so far is documented under "What didn't work."

---

## Architecture

```
            ┌────────────────────────────────────────────┐
            │  physics/  (C++17, no graphics)            │
            │  RigidBody · World · Thrusters · Legs      │
            │  Contact · Drag · Wind · LandingEnv        │
            │  Baseline                                  │
            └───────┬──────────────────────┬─────────────┘
                    │                      │
          ┌─────────▼────────┐    ┌────────▼────────────┐
          │ bindings/        │    │ viewer/             │
          │ pybind11 module  │    │ SDL2 + OpenGL       │
          └─────────┬────────┘    └─────────────────────┘
                    │
          ┌─────────▼────────────────────────────────────┐
          │ python/                                      │
          │ rocket_env.py (Gymnasium) → train.py (PPO)   │
          │ check_policy.py · compare_returns.py         │
          └──────────────────────────────────────────────┘
```

The physics library has no dependency on SDL, OpenGL, ImGui, or Python. The same library drives the viewer, the tests, and the RL environment.

---

## Repository layout

```
physics/     Physics library, environment, baseline (rocketphys)
tests/       GoogleTest suite: physics, thrusters, contact, drag, wind, legs, environment
viewer/      SDL2/OpenGL viewer
bindings/    pybind11 module (rocketphys_py)
python/      Gymnasium wrapper, training, evaluation scripts
```

---

## Build

### C++ (Windows, Visual Studio 2022)

1. Install Visual Studio 2022 with the "Desktop development with C++" workload and the CMake tools.
2. Install vcpkg, clone it outside Program Files, and run its bootstrap script. Set the `VCPKG_ROOT` environment variable to its folder.
3. Install Python 3.13 from python.org, and note the path to `python.exe`.
4. In `CMakePresets.json`, set the toolchain path to your vcpkg folder and `Python_EXECUTABLE` to your Python 3.13 path.
5. Open the project folder in Visual Studio (File → Open → Folder). Choose **Project → Delete Cache and Reconfigure**. The first configure downloads and builds dependencies, which takes several minutes.
6. Select **x64 Release** and choose **Build → Build All**. The Python module is copied into `python/`.

Use Release for any benchmark or training run. Debug is about 19 times slower.

### Python

From the `python` folder:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.\.venv\Scripts\python.exe` directly.

---

## Running

### Tests
Run the suite from Visual Studio's Test Explorer, or build and run `physics_tests`.

### Viewer
| Key | Action |
|---|---|
| H | Hover on or off |
| M | Manual thruster test mode |
| 1–5 (manual mode, hold) | Main engine, pitch A, pitch B, roll A, roll B |
| T | Tilt the rocket 10° |
| R | Reset |
| V | Show or hide thrust arrows |
| A / B / G | Atmosphere, base wind, gusts |
| Q / E | Orbit camera |
| Esc | Quit |

### Python checks
```
python check_rocket_env.py
python test_binding.py
```

### Training
```
python train.py --smoke            # 50k-step check run
python train.py --name myrun       # main run, 3M steps by default
tensorboard --logdir runs
```

### Evaluation
```
python check_policy.py runs/<run-folder>
python compare_returns.py
```

---

## Environment definition

**Action** (`[throttle, pitch, roll]`, each in [−1, 1])
- Throttle maps to the main engine as (a + 1) / 2. Hover is about −0.215.
- Pitch: positive fires pitch pair B, negative fires pitch pair A.
- Roll: positive fires roll pair A, negative fires roll pair B.

**Observation** (18 values, 21 with the wind sensor)

| Index | Value | Scale |
|---|---|---|
| 0 | Altitude of the lowest point | /100 |
| 1–3 | Velocity | /50 |
| 4–5 | Horizontal position | /100 |
| 6–14 | Rotation matrix | none |
| 15–17 | Angular velocity | /10 |
| 18–20 | Air-relative velocity (wind sensor only) | /50 |

**Reward**
- Per step: −0.01 base cost, tilt cost up to −0.05, spin cost, speed-envelope cost for falling faster than √h (minimum 1.5 m/s).
- Potential-based shaping: +0.05 per metre of altitude lost.
- Terminal: +100 on landing, minus 5 per m/s of touchdown speed and 0.2 per degree of tilt. −100 on crash. −50 on timeout.

**Termination:** landed, crashed, or 60 s of simulated time.

**Difficulty:** 0 starts upright and still. 1 starts with a random orientation and velocity. Starting altitude is 15–200 m.

---

## What didn't work

These failures are part of the project's record, and each one taught something about the environment.

1. **Hovering exploit (original reward).** A policy that avoids crashing earns about −20 per episode by hovering until the time limit. Landing earns +100 only after a long, uncertain descent. Early in training, hovering was the reward-maximizing choice, and the policy converged to it. Evaluation confirmed 50 of 50 timeouts.
2. **Discount too short.** With γ = 0.995, the effective horizon was about 7 s, which is too short to value a landing from 200 m. Raising γ to 0.999 helped the policy see the landing bonus, but did not alone make it descend.
3. **Descent penalty discouraged the approach.** Penalizing fall speed below 30 m made descending expensive. Replacing it with potential-based altitude shaping fixed that direction, but the policy then descended too fast. All 16 crashes in that run were too fast, with touchdown speeds from 5.5 to 14 m/s, and none were too tilted.
4. **Speed envelope reverted to hovering.** Penalizing speed above √h made fast descents costly while they were happening. That made hovering the cheapest safe option again, and evaluation returned to 50 of 50 timeouts.

Likely next steps are a curriculum that starts at low altitude, so the policy meets landings it can reach, and a comparison of reward variants on the same seeds. These are in progress.

---

## Known limitations

- **No learned policy has landed reliably.** The baseline is the only controller with a measured landing rate.
- **No yaw control.** No thruster produces torque about the long axis, so the rocket can't turn about Y. Landing doesn't depend on heading, so this was left for future work.
- **Fixed center of pressure.** Real aerodynamic centers move with angle of attack and speed. This model uses one point.
- **No leg compliance.** Feet are rigid point contacts, so landings are stiffer than they would be with real struts.
- **Friction is a single coefficient.** There is no separate static and kinetic friction.
- **Perfect sensing.** The observation is noise-free and has no delay. A policy trained this way may not transfer directly to real sensors.
- **Inverted low starts are not recoverable.** Physics limits how far an inverted rocket can fall before RCS can recover it.
- **Windows-only build instructions.** The C++ build has been tested only with Visual Studio on Windows.

---

## Roadmap

- Curriculum that starts at low altitude, with difficulty raised as landing rate improves.
- Reward variants compared on fixed seeds, with the baseline as the reference.
- Wind and sensor-noise robustness tests on trained policies.
- Legs as a second experiment, compared against the no-legs policy.
- Yaw thrusters.
- Pressure-center movement with angle of attack.
- Tangential yaw control and a 1000 m climb-and-return mission.

---

## Reproducibility

- Physics uses a fixed timestep, so the same seed and actions give the same trajectory.
- Each training run saves the model and its VecNormalize statistics together. Evaluation must use both.
- Pinned Python packages are in `requirements.txt`.
- Results are tagged in git. Each run folder records the configuration it used.
