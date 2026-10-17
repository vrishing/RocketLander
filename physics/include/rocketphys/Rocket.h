#pragma once
#include <cstddef>

#include <glm/glm.hpp>

#include "rocketphys/RigidBody.h"

namespace rocketphys {

    // Thruster indices, in the order MakeRocketBody() adds them.
    inline constexpr std::size_t kMainEngine = 0;
    inline constexpr std::size_t kPitchATop = 1;
    inline constexpr std::size_t kPitchABottom = 2;
    inline constexpr std::size_t kPitchBTop = 3;
    inline constexpr std::size_t kPitchBBottom = 4;
    inline constexpr std::size_t kRollATop = 5;
    inline constexpr std::size_t kRollABottom = 6;
    inline constexpr std::size_t kRollBTop = 7;
    inline constexpr std::size_t kRollBBottom = 8;

    // 1000 kg cylinder, radius 0.5 m, height 8 m, centered on the origin.
    RigidBody MakeRocketBody();

    // Attitude controller gains.
    struct AttitudeGains {
        float kp = 2644.0f;  // N*m per radian of tilt   (I * w^2, with I = 5396, w = 0.7 rad/s)
        float kd = 5292.0f;  // N*m*s per rad/s          (2 * zeta * I * w, with zeta = 0.7)
    };

    // Tilt error in world space: up axis crossed with world up. Its direction is the
    // rotation axis that brings the rocket upright, and its length is about sin(tilt).
    glm::vec3 RocketTiltError(const RigidBody& rocket);

    // Fires the RCS diagonal pairs to drive the rocket upright. Call once per physics
    // step, before world.Step(). Leaves the main engine alone.
    void UpdateRocketAttitude(RigidBody& rocket, const AttitudeGains& gains = AttitudeGains{});

}  // namespace rocketphys