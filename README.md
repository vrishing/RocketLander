# Rocket Lander

C++ physics library + viewer. Later: Python bindings and an RL agent that learns to land.

## Layout
- `physics/`  static library `rocketphys` (RigidBody, World). Depends on GLM only.
- `tests/`    GoogleTest unit tests for the physics library.
- `viewer/`   SDL2 + OpenGL + ImGui app. The only place graphics code is allowed.

## Build (Windows, Visual Studio 2022)
1. Install vcpkg and set the `VCPKG_ROOT` environment variable.
2. File > Open > Folder, choose this directory, pick the `x64 Debug` preset.
3. Select `viewer.exe` (or `physics_tests`) as the startup item and run.

## Controls
W/S thrust, A/D turn, Space jump, R reset, Q/E orbit camera, mouse wheel zoom, Esc quit.
