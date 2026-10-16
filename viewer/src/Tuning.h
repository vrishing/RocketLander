#pragma once
// Values you can change live from the ImGui panel. Plain data, no logic.

struct Tuning {
    float gravity     = 9.81f;   // m/s^2, magnitude (pulls toward -Y)
    float restitution = 0.30f;   // 0 = no bounce, 1 = perfectly elastic
    float mass        = 5.0f;    // kg
    float thrust      = 60.0f;   // N, W/S key force along the body's forward axis
    float turnPower   = 4.0f;    // N*m, A/D key torque about the Y axis
    float jumpSpeed   = 6.0f;    // m/s, instant upward velocity added by Space

    static Tuning Defaults() { return Tuning{}; }
};
