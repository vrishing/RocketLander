#pragma once
// Leg: one landing foot. Plain data plus the deployment limits.

#include <glm/glm.hpp>

namespace rocketphys {

    struct Leg {
        glm::vec3 localAttach{ 0.0f };  // where the leg joins the body, body axes, m
        glm::vec3 localFoot{ 0.0f };    // where the foot sits, body axes, m
        bool deployed = false;        // latched: once true, stays true until ResetLegs()
    };

    // Deployment rule. Legs come out when the base is below this altitude...
    inline constexpr float kLegDeployAltitude = 15.0f;  // m
    // ...and the body is falling faster than this.
    inline constexpr float kLegDeploySpeed = 1.0f;      // m/s

}  // namespace rocketphys