#pragma once
#include <stdexcept>

#include <glm/glm.hpp>

namespace rocketphys {

    struct Thruster {
        glm::vec3 localPosition{ 0.0f };    // meters, relative to center of mass, body axes
        glm::vec3 localDirection{ 0.0f };   // unit vector, body axes, direction the exhaust pushes
        float maxForce = 0.0f;            // newtons
        float throttle = 0.0f;            // 0 to 1, clamped by RigidBody::SetThrottle

        Thruster(const glm::vec3& position, const glm::vec3& direction, float force)
        : localPosition(position), maxForce(force) {
            if (glm::length(direction) == 0.0f) {
                throw std::invalid_argument("Thruster direction must not be zero");
            }
            localDirection = glm::normalize(direction);
        }
    };

}  // namespace rocketphys