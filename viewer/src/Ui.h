#pragma once
// Ui: wraps Dear ImGui so no other file needs to include ImGui headers.

#include <SDL.h>

#include "Tuning.h"

struct HudStats {
    float height = 0.0f;       // m
    float speed = 0.0f;        // m/s
    float angularSpeed = 0.0f; // rad/s
    double simTime = 0.0;      // s
    float fps = 0.0f;
};

struct UiActions {
    bool resetBody = false;
    bool resetParams = false;
};

class Ui {
public:
    void Init(SDL_Window* window, SDL_GLContext glContext);
    void Shutdown();

    void ProcessEvent(const SDL_Event& e);
    bool WantsKeyboard() const;  // true while typing in an ImGui widget
    bool WantsMouse() const;

    void BeginFrame();
    UiActions DrawPanel(Tuning& tuning, const HudStats& stats);
    void EndFrame();  // renders the ImGui draw data
};
