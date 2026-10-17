#pragma once

#include <cstddef>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "rocketphys/Thruster.h"

namespace rocketphys {

    class RigidBody {
    public:
        // A mass <= 0 makes the body static (invMass_ = 0, it never moves).
        void SetMass(float mass);

        // Call AFTER SetMass(): inertia depends on mass.
        void SetBoxInertia(float width, float height, float depth);

        // Call AFTER SetMass(). Long axis is Y.
        void SetCylinderInertia(float radius, float height);

        // Retention factor per 1/60 s. 1.0 = no damping.
        void SetDamping(float linear, float angular);

        // Distance from the body's center to its lowest point, used for ground contact.
        void SetHalfHeight(float halfHeight) { halfHeight_ = halfHeight; }

        // ---------- Per-step API ----------
        void ApplyForce(const glm::vec3& worldForce);
        void ApplyTorque(const glm::vec3& worldTorque);
        void ApplyForceAtPoint(const glm::vec3& worldForce, const glm::vec3& worldPoint);

        // Advance by dt seconds, then clear the accumulators.
        void Integrate(float dt);

        glm::mat3 GetInvInertiaBody() const;


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

        // ---------- Thrusters ----------
        // Returns the index of the new thruster (0 for the first one added).
        std::size_t AddThruster(const Thruster& t);

        // Sets a thruster's throttle, clamped to [0, 1]. Throws if index is out of range.
        void SetThrottle(std::size_t index, float value);

        // Converts every thruster's force into a world-space force and torque.
        // Call once per physics step, before Integrate().
        void ApplyThrusters();

        const std::vector<Thruster>& GetThrusters() const { return thrusters_; }

    private:
        // Linear state
        glm::vec3 position_{ 0.0f };
        glm::vec3 velocity_{ 0.0f };
        float mass_ = 1.0f;
        float invMass_ = 1.0f;
        glm::vec3 force_{ 0.0f };

        // Rotational state
        glm::quat orientation_{ 1.0f, 0.0f, 0.0f, 0.0f };  // identity (w, x, y, z)
        glm::vec3 angularVelocity_{ 0.0f };                // world space, rad/s
        glm::mat3 invInertiaBody_{ 1.0f };                 // inverse inertia in body space
        glm::vec3 torque_{ 0.0f };

        // Tuning
        float linearDamping_ = 1.0f;
        float angularDamping_ = 1.0f;
        float halfHeight_ = 0.5f;

        std::vector<Thruster> thrusters_;
    };

}  // namespace rocketphys