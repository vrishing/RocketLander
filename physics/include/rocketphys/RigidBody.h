#pragma once
// RigidBody: one simulated object. Pure physics, no rendering, no SDL, no OpenGL.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace rocketphys {

class RigidBody {
public:
    // ---------- Setup (call once, before simulating) ----------
    // A mass <= 0 makes the body static (invMass_ = 0, it never moves).
    void SetMass(float mass);

    // Call AFTER SetMass(): inertia depends on mass.
    // Ixx = m/12 * (h^2 + d^2), Iyy = m/12 * (w^2 + d^2), Izz = m/12 * (w^2 + h^2)
    void SetBoxInertia(float width, float height, float depth);

    // Retention factor per 1/60 s. 1.0 = no damping, 0.98 = lose 2% each 1/60 s.
    // Applied as factor^(dt * 60) so behavior is independent of the timestep.
    void SetDamping(float linear, float angular);

    // Distance from the body's center to its lowest point, used for ground contact.
    void SetHalfHeight(float halfHeight) { halfHeight_ = halfHeight; }

    // ---------- Per-step API ----------
    // Forces/torques are in WORLD space and accumulate until Integrate() clears them.
    void ApplyForce(const glm::vec3& worldForce);
    void ApplyTorque(const glm::vec3& worldTorque);
    // Adds the force AND the torque (r x F) produced by pushing at worldPoint.
    void ApplyForceAtPoint(const glm::vec3& worldForce, const glm::vec3& worldPoint);

    // Advance by dt seconds, then clear the accumulators.
    void Integrate(float dt);

    // ---------- Accessors ----------
    const glm::vec3& GetPosition() const { return position_; }
    const glm::vec3& GetVelocity() const { return velocity_; }
    const glm::quat& GetOrientation() const { return orientation_; }
    const glm::vec3& GetAngularVelocity() const { return angularVelocity_; }
    float GetMass() const { return mass_; }
    float GetInvMass() const { return invMass_; }
    float GetHalfHeight() const { return halfHeight_; }
    glm::mat3 GetRotationMatrix() const { return glm::mat3_cast(orientation_); }

    void SetPosition(const glm::vec3& p) { position_ = p; }
    void SetVelocity(const glm::vec3& v) { velocity_ = v; }
    void SetOrientation(const glm::quat& q) { orientation_ = glm::normalize(q); }
    void SetAngularVelocity(const glm::vec3& w) { angularVelocity_ = w; }

    // Zero velocities and accumulators (position/orientation are set separately).
    void ZeroMotion();

private:
    // Linear state
    glm::vec3 position_{0.0f};
    glm::vec3 velocity_{0.0f};
    float mass_ = 1.0f;
    float invMass_ = 1.0f;
    glm::vec3 force_{0.0f};

    // Rotational state
    glm::quat orientation_{1.0f, 0.0f, 0.0f, 0.0f};  // identity (w, x, y, z)
    glm::vec3 angularVelocity_{0.0f};                // world space, rad/s
    glm::mat3 invInertiaBody_{1.0f};                 // inverse inertia in body space
    glm::vec3 torque_{0.0f};

    // Tuning
    float linearDamping_ = 1.0f;
    float angularDamping_ = 1.0f;
    float halfHeight_ = 0.5f;
};

}  // namespace rocketphys
