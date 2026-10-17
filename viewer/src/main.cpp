// Viewer: window, input, camera, and the fixed-timestep loop.
// Physics lives in physics/ (no graphics). Drawing lives in Renderer.

#include <GL/glew.h>
#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Renderer.h"
#include "Tuning.h"
#include "rocketphys/World.h"

namespace {

    constexpr float kBoxSize = 1.0f;  // the cube is 1 x 1 x 1 m
    const glm::vec3 kStartPosition{ 0.0f, 5.0f, 0.0f };

    // Push the tuning values into the physics world and body.
    void ApplyTuning(rocketphys::World& world, std::size_t id, const Tuning& t) {
        world.SetGravity({ 0.0f, -t.gravity, 0.0f });
        world.SetRestitution(t.restitution);

        rocketphys::RigidBody& body = world.GetBody(id);
        body.SetMass(t.mass);  // must come before inertia
        body.SetBoxInertia(kBoxSize, kBoxSize, kBoxSize);
    }

    void ResetBody(rocketphys::World& world, std::size_t id) {
        rocketphys::RigidBody& body = world.GetBody(id);
        body.SetPosition(kStartPosition);
        body.SetOrientation(glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
        body.ZeroMotion();
    }

    bool IsGrounded(const rocketphys::RigidBody& body) {
        return body.GetPosition().y - body.GetHalfHeight() <= 0.02f;
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
    Tuning tuning = Tuning::Defaults();
    rocketphys::World world;

    rocketphys::RigidBody cube;
    cube.SetDamping(0.995f, 0.97f);
    cube.SetHalfHeight(kBoxSize * 0.5f);
    const std::size_t cubeId = world.AddBody(cube);
    ApplyTuning(world, cubeId, tuning);
    ResetBody(world, cubeId);

    // ---------- Camera (orbits the cube) ----------
    float camYaw = 0.6f;
    float camPitch = 0.45f;
    float camDistance = 14.0f;

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
                case SDLK_r: ResetBody(world, cubeId); break;
                case SDLK_SPACE: {
                    rocketphys::RigidBody& b = world.GetBody(cubeId);
                    if (IsGrounded(b)) {
                        glm::vec3 v = b.GetVelocity();
                        v.y += tuning.jumpSpeed;
                        b.SetVelocity(v);
                    }
                    break;
                }
                default: break;
                }
            }
            if (e.type == SDL_MOUSEWHEEL) {
                camDistance = std::clamp(camDistance - static_cast<float>(e.wheel.y), 4.0f, 40.0f);
            }
        }

        // ---- Timing ----
        const Uint64 now = SDL_GetPerformanceCounter();
        const double frameTime = static_cast<double>(now - previous) / freq;
        previous = now;
        accumulator += std::min(frameTime, 0.25);  // clamp: avoids a spiral of death after a stall

        // ---- Held keys (read once per frame, applied every physics step) ----
        const Uint8* keys = SDL_GetKeyboardState(nullptr);
        const bool w = keys[SDL_SCANCODE_W];
        const bool s = keys[SDL_SCANCODE_S];
        const bool a = keys[SDL_SCANCODE_A];
        const bool d = keys[SDL_SCANCODE_D];
        const float orbit = 1.5f * static_cast<float>(frameTime);
        if (keys[SDL_SCANCODE_Q]) camYaw -= orbit;
        if (keys[SDL_SCANCODE_E]) camYaw += orbit;

        // ---- Fixed-timestep physics ----
        ApplyTuning(world, cubeId, tuning);
        while (accumulator >= rocketphys::World::kFixedDt) {
            rocketphys::RigidBody& body = world.GetBody(cubeId);
            const glm::vec3 forward = body.GetOrientation() * glm::vec3(0.0f, 0.0f, -1.0f);
            if (w) body.ApplyForce(forward * tuning.thrust);
            if (s) body.ApplyForce(-forward * tuning.thrust);
            if (a) body.ApplyTorque({ 0.0f, tuning.turnPower, 0.0f });
            if (d) body.ApplyTorque({ 0.0f, -tuning.turnPower, 0.0f });

            world.Step();
            accumulator -= rocketphys::World::kFixedDt;
        }

        // ---- Camera matrices ----
        const rocketphys::RigidBody& body = world.GetBody(cubeId);
        int drawW = 1, drawH = 1;
        SDL_GL_GetDrawableSize(window, &drawW, &drawH);
        glViewport(0, 0, drawW, drawH);
        const float aspect = static_cast<float>(drawW) / static_cast<float>(std::max(drawH, 1));

        const glm::vec3 target = body.GetPosition();
        const glm::vec3 eye = target + camDistance * glm::vec3(
            std::sin(camYaw) * std::cos(camPitch),
            std::sin(camPitch),
            std::cos(camYaw) * std::cos(camPitch));
        const glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 200.0f);
        const glm::mat4 viewProj = proj * view;

        // ---- Draw ----
        glClearColor(0.07f, 0.09f, 0.13f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        renderer.DrawGrid(viewProj);
        const glm::mat4 model = glm::translate(glm::mat4(1.0f), body.GetPosition()) *
            glm::mat4_cast(body.GetOrientation());
        renderer.DrawBody(viewProj, model);

        SDL_GL_SwapWindow(window);
    }

    renderer.Shutdown();
    SDL_GL_DeleteContext(glContext);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}