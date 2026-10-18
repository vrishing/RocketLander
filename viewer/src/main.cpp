// Viewer: window, input, camera, and the fixed-timestep loop.
// Physics lives in physics/ (no graphics). Drawing lives in Renderer.

#include <GL/glew.h>
#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Renderer.h"
#include "rocketphys/Rocket.h"
#include "rocketphys/World.h"

namespace {

    // Rocket geometry. Must match MakeRocketBody() in Rocket.cpp.
    constexpr float kRocketRadius = 0.5f;
    constexpr float kRocketHeight = 8.0f;

    // Hover controller. Same gains as the hover test.
    constexpr float kHoverThrottle = 9810.0f / 25000.0f;  // weight / max thrust
    constexpr float kHoverKp = 0.002f;
    constexpr float kHoverKd = 0.02f;
    constexpr float kTargetY = 20.0f;                     // body center target, m
    const glm::vec3 kStartPosition{ 0.0f, 30.0f, 0.0f };

    // Display only: meters of arrow per newton of thrust. Physics is unaffected.
    constexpr float kThrustDisplayScale = 0.0002f;

    // Manual test mode: each key fires one group of thrusters while held.
    struct ThrusterGroup {
        SDL_Scancode key;
        const char* name;
        std::vector<std::size_t> thrusters;
    };

    const std::vector<ThrusterGroup> kThrusterGroups = {
        { SDL_SCANCODE_1, "main",{ rocketphys::kMainEngine } },
        { SDL_SCANCODE_2, "pitchA",{ rocketphys::kPitchATop, rocketphys::kPitchABottom } },
        { SDL_SCANCODE_3, "pitchB",{ rocketphys::kPitchBTop, rocketphys::kPitchBBottom } },
        { SDL_SCANCODE_4, "rollA",{ rocketphys::kRollATop, rocketphys::kRollABottom } },
        { SDL_SCANCODE_5, "rollB",{ rocketphys::kRollBTop, rocketphys::kRollBBottom } },
    };

    void ResetRocket(rocketphys::World& world, std::size_t id) {
        rocketphys::RigidBody& rocket = world.GetBody(id);
        rocket.SetPosition(kStartPosition);
        rocket.SetOrientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        rocket.ZeroMotion();
        rocket.ResetLandingStatus();
    }

    // Same start, but tilted about the X axis.
    void ResetRocketTilted(rocketphys::World& world, std::size_t id, float degrees) {
        rocketphys::RigidBody& rocket = world.GetBody(id);
        rocket.SetPosition(kStartPosition);
        rocket.SetOrientation(glm::angleAxis(glm::radians(degrees), glm::vec3(1.0f, 0.0f, 0.0f)));
        rocket.ZeroMotion();
        rocket.ResetLandingStatus();
    }

    // Returns the main-engine throttle for this physics step.
    float HoverThrottle(const rocketphys::RigidBody& rocket) {
        const float error = kTargetY - rocket.GetPosition().y;
        const float vy = rocket.GetVelocity().y;
        return kHoverThrottle + kHoverKp * error - kHoverKd * vy;
    }

    // Manual mode: each group is on while its key is held, off otherwise.
    // The groups cover all nine thrusters, so every throttle is set each step.
    void ApplyManualThrottles(rocketphys::RigidBody& rocket, const Uint8* keys) {
        for (const ThrusterGroup& g : kThrusterGroups) {
            const float value = keys[g.key] ? 1.0f : 0.0f;
            for (std::size_t index : g.thrusters) {
                rocket.SetThrottle(index, value);
            }
        }
    }

    // Automatic mode: only the main engine is driven, so RCS is switched off.
    void ClearRcsThrottles(rocketphys::RigidBody& rocket) {
        const std::size_t count = rocket.GetThrusters().size();
        for (std::size_t i = 0; i < count; ++i) {
            if (i != rocketphys::kMainEngine) rocket.SetThrottle(i, 0.0f);
        }
    }

    // Builds arrow line segments for every thruster that is firing.
    void BuildThrustLines(const rocketphys::RigidBody& rocket,
        std::vector<glm::vec3>& mainLines,
        std::vector<glm::vec3>& rcsLines) {
        mainLines.clear();
        rcsLines.clear();

        const glm::mat3 R = rocket.GetRotationMatrix();
        const std::vector<rocketphys::Thruster>& thrusters = rocket.GetThrusters();

        for (std::size_t i = 0; i < thrusters.size(); ++i) {
            const rocketphys::Thruster& t = thrusters[i];
            if (t.throttle <= 0.0f) continue;

            // Same world-space point and direction that ApplyThrusters() uses.
            const glm::vec3 start = rocket.GetPosition() + R * t.localPosition;
            const glm::vec3 dir = R * t.localDirection;
            const float length = t.throttle * t.maxForce * kThrustDisplayScale;

            std::vector<glm::vec3>& target = (i == rocketphys::kMainEngine) ? mainLines : rcsLines;
            target.push_back(start);
            target.push_back(start + dir * length);
        }
    }

    void PrintControls() {
        std::printf(
            "Controls:\n"
            "  H  hover on/off (automatic mode)\n"
            "  M  manual thruster test mode on/off\n"
            "  1-5 (manual mode, hold): main, pitchA, pitchB, rollA, rollB\n"
            "  T  tilt rocket 10 degrees   R  reset   V  thrust arrows   Q/E  orbit   Esc  quit\n");
    }

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
    // ---------- Window and OpenGL context ----------
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    SDL_Window* window = SDL_CreateWindow(
        "Rocket Lander", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }
    SDL_GLContext glContext = SDL_GL_CreateContext(window);
    if (!glContext) {
        std::fprintf(stderr, "SDL_GL_CreateContext failed: %s\n", SDL_GetError());
        return 1;
    }
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "glewInit failed\n");
        return 1;
    }
    SDL_GL_SetSwapInterval(1);  // vsync
    glEnable(GL_DEPTH_TEST);

    Renderer renderer;
    if (!renderer.Init()) return 1;

    // ---------- Simulation ----------
    rocketphys::World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });
    const std::size_t rocketId = world.AddBody(rocketphys::MakeRocketBody());
    ResetRocket(world, rocketId);
    bool hoverOn = true;
    bool manualMode = false;
    bool showThrust = true;

    // Reused every frame so we don't reallocate.
    std::vector<glm::vec3> mainLines;
    std::vector<glm::vec3> rcsLines;

    // ---------- Camera (orbits the rocket) ----------
    float camYaw = 0.6f;
    float camPitch = 0.2f;
    float camDistance = 35.0f;

    PrintControls();

    // ---------- Main loop ----------
    double accumulator = 0.0;
    Uint64 previous = SDL_GetPerformanceCounter();
    const double freq = static_cast<double>(SDL_GetPerformanceFrequency());
    bool running = true;

    while (running) {
        // ---- Events ----
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;

            if (e.type == SDL_KEYDOWN && !e.key.repeat) {
                switch (e.key.keysym.sym) {
                case SDLK_ESCAPE: running = false; break;
                case SDLK_r: ResetRocket(world, rocketId); break;
                case SDLK_t: ResetRocketTilted(world, rocketId, 10.0f); break;
                case SDLK_h:
                    hoverOn = !hoverOn;
                    std::printf("Hover %s\n", hoverOn ? "ON" : "OFF");
                    break;
                case SDLK_m:
                    manualMode = !manualMode;
                    std::printf("Manual thruster test %s\n", manualMode ? "ON" : "OFF");
                    break;
                case SDLK_v:
                    showThrust = !showThrust;
                    std::printf("Thrust vectors %s\n", showThrust ? "ON" : "OFF");
                    break;
                default: break;
                }
            }
            if (e.type == SDL_MOUSEWHEEL) {
                camDistance = std::clamp(camDistance - 2.0f * static_cast<float>(e.wheel.y),
                    10.0f, 150.0f);
            }
        }

        // ---- Timing ----
        const Uint64 now = SDL_GetPerformanceCounter();
        const double frameTime = static_cast<double>(now - previous) / freq;
        previous = now;
        accumulator += std::min(frameTime, 0.25);  // clamp: avoids a spiral of death after a stall

        // ---- Keyboard state (read once per frame) ----
        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        const float orbit = 1.5f * static_cast<float>(frameTime);
        if (keys[SDL_SCANCODE_Q]) camYaw -= orbit;
        if (keys[SDL_SCANCODE_E]) camYaw += orbit;

        // ---- Fixed-timestep physics ----
        while (accumulator >= rocketphys::World::kFixedDt) {
            rocketphys::RigidBody& rocket = world.GetBody(rocketId);

            float mainThrottle = 0.0f;
            if (manualMode) {
                ApplyManualThrottles(rocket, keys);
                mainThrottle = rocket.GetThrusters()[rocketphys::kMainEngine].throttle;
            }
            else {
                mainThrottle = hoverOn ? HoverThrottle(rocket) : 0.0f;
                rocket.SetThrottle(rocketphys::kMainEngine, mainThrottle);
                ClearRcsThrottles(rocket);
            }

            world.Step();
            accumulator -= rocketphys::World::kFixedDt;

            const rocketphys::RigidBody& r = world.GetBody(rocketId);

            // Manual mode: print angular velocity every 0.25 s, for checking signs.
            if (manualMode && world.GetStepCount() % 30 == 0) {
                const glm::vec3 w = r.GetAngularVelocity();
                std::printf("t=%6.2f  y=%8.3f  vy=%7.3f  w=(%7.3f, %7.3f, %7.3f)  tilt=%6.2f deg\n",
                    world.GetTime(), r.GetPosition().y, r.GetVelocity().y,
                    w.x, w.y, w.z, r.GetTiltDegrees());
            }

            // Automatic mode: once per simulated second.
            if (!manualMode && world.GetStepCount() % 120 == 0) {
                std::printf("t=%6.2f  y=%8.3f  vy=%8.3f  throttle=%.3f\n",
                    world.GetTime(), r.GetPosition().y, r.GetVelocity().y, mainThrottle);
            }
        }

        // ---- Camera matrices ----
        const rocketphys::RigidBody& rocket = world.GetBody(rocketId);
        int drawW = 1, drawH = 1;
        SDL_GL_GetDrawableSize(window, &drawW, &drawH);
        glViewport(0, 0, drawW, drawH);
        const float aspect = static_cast<float>(drawW) / static_cast<float>(std::max(drawH, 1));

        const glm::vec3 target = rocket.GetPosition();
        const glm::vec3 eye = target + camDistance * glm::vec3(
            std::sin(camYaw) * std::cos(camPitch),
            std::sin(camPitch),
            std::cos(camYaw) * std::cos(camPitch));
        const glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 500.0f);
        const glm::mat4 viewProj = proj * view;

        // ---- Draw ----
        glClearColor(0.07f, 0.09f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderer.DrawGrid(viewProj);

        const glm::mat4 model =
            glm::translate(glm::mat4(1.0f), rocket.GetPosition()) *
            glm::mat4_cast(rocket.GetOrientation()) *
            glm::scale(glm::mat4(1.0f), glm::vec3(2.0f * kRocketRadius, kRocketHeight, 2.0f * kRocketRadius));
        renderer.DrawBody(viewProj, model);

        if (showThrust) {
            BuildThrustLines(rocket, mainLines, rcsLines);
            renderer.DrawLines(viewProj, mainLines, glm::vec3(1.0f, 0.6f, 0.1f));
            renderer.DrawLines(viewProj, rcsLines, glm::vec3(0.3f, 0.8f, 1.0f));
        }

        SDL_GL_SwapWindow(window);
    }

    renderer.Shutdown();
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}