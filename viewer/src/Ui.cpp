#include "Ui.h"

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>

void Ui::Init(SDL_Window* window, SDL_GLContext glContext) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window, glContext);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Ui::Shutdown() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
}

void Ui::ProcessEvent(const SDL_Event& e) { ImGui_ImplSDL2_ProcessEvent(&e); }
bool Ui::WantsKeyboard() const { return ImGui::GetIO().WantCaptureKeyboard; }
bool Ui::WantsMouse() const { return ImGui::GetIO().WantCaptureMouse; }

void Ui::BeginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
}

UiActions Ui::DrawPanel(Tuning& t, const HudStats& s) {
    UiActions actions;

    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::Begin("Physics tuning");

    ImGui::SliderFloat("Gravity (m/s^2)", &t.gravity, 0.0f, 20.0f);
    ImGui::SliderFloat("Restitution", &t.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Mass (kg)", &t.mass, 0.1f, 50.0f);
    ImGui::SliderFloat("Thrust (N)", &t.thrust, 0.0f, 300.0f);
    ImGui::SliderFloat("Turn power (N*m)", &t.turnPower, 0.0f, 20.0f);
    ImGui::SliderFloat("Jump speed (m/s)", &t.jumpSpeed, 0.0f, 20.0f);

    if (ImGui::Button("Reset body (R)")) actions.resetBody = true;
    ImGui::SameLine();
    if (ImGui::Button("Reset parameters")) actions.resetParams = true;

    ImGui::Separator();
    ImGui::Text("Height:        %.2f m", s.height);
    ImGui::Text("Speed:         %.2f m/s", s.speed);
    ImGui::Text("Angular speed: %.2f rad/s", s.angularSpeed);
    ImGui::Text("Sim time:      %.1f s", s.simTime);
    ImGui::Text("FPS:           %.0f", s.fps);
    ImGui::Separator();
    ImGui::TextUnformatted("W/S thrust   A/D turn   Space jump");
    ImGui::TextUnformatted("Q/E orbit camera   wheel zoom   Esc quit");

    ImGui::End();
    return actions;
}

void Ui::EndFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
