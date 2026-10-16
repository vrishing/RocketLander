#include "rocketphys/RigidBody.h"

#include <cmath>

namespace rocketphys {

void RigidBody::SetMass(float mass) {
    mass_ = mass;
    invMass_ = (mass > 0.0f) ? 1.0f / mass : 0.0f;
}

void RigidBody::SetBoxInertia(float width, float height, float depth) {
    invInertiaBody_ = glm::mat3(0.0f);  // all zeros (static bodies stay at zero)
    if (invMass_ == 0.0f) return;

    const float k = mass_ / 12.0f;
    const float ixx = k * (height * height + depth * depth);
    const float iyy = k * (width * width + depth * depth);
    const float izz = k * (width * width + height * height);

    // glm matrices are indexed [column][row]; for a diagonal matrix it doesn't matter.
    invInertiaBody_[0][0] = 1.0f / ixx;
    invInertiaBody_[1][1] = 1.0f / iyy;
    invInertiaBody_[2][2] = 1.0f / izz;
}

void RigidBody::SetDamping(float linear, float angular) {
    linearDamping_ = linear;
    angularDamping_ = angular;
}

void RigidBody::ApplyForce(const glm::vec3& worldForce) { force_ += worldForce; }

void RigidBody::ApplyTorque(const glm::vec3& worldTorque) { torque_ += worldTorque; }

void RigidBody::ApplyForceAtPoint(const glm::vec3& worldForce, const glm::vec3& worldPoint) {
    // TODO (hour 1.5): assumes center of mass == position_. Revisit once you add a
    // center-of-mass offset for the rocket.
    force_ += worldForce;
    torque_ += glm::cross(worldPoint - position_, worldForce);
}

void RigidBody::ZeroMotion() {
    velocity_ = glm::vec3(0.0f);
    angularVelocity_ = glm::vec3(0.0f);
    force_ = glm::vec3(0.0f);
    torque_ = glm::vec3(0.0f);
}

void RigidBody::Integrate(float dt) {
    if (invMass_ == 0.0f) {  // static body
        force_ = glm::vec3(0.0f);
        torque_ = glm::vec3(0.0f);
        return;
    }

    // ---- Linear: semi-implicit Euler (update velocity first, then position) ----
    velocity_ += force_ * invMass_ * dt;
    velocity_ *= std::pow(linearDamping_, dt * 60.0f);
    position_ += velocity_ * dt;

    // ---- Angular ----
    // World-space inverse inertia: R * Ibody^-1 * R^T
    const glm::mat3 R = glm::mat3_cast(orientation_);
    const glm::mat3 invInertiaWorld = R * invInertiaBody_ * glm::transpose(R);

    angularVelocity_ += invInertiaWorld * torque_ * dt;
    angularVelocity_ *= std::pow(angularDamping_, dt * 60.0f);

    // dq/dt = 0.5 * (0, w) * q, then renormalize to fight numerical drift.
    const glm::quat omegaQuat(0.0f, angularVelocity_.x, angularVelocity_.y, angularVelocity_.z);
    orientation_ += (omegaQuat * orientation_) * (0.5f * dt);
    orientation_ = glm::normalize(orientation_);

    force_ = glm::vec3(0.0f);
    torque_ = glm::vec3(0.0f);
}

}  // namespace rocketphys
