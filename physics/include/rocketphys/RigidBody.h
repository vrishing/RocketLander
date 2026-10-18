#pragma once
// RigidBody: one simulated object. Pure physics, no rendering, no SDL, no OpenGL.

#include <cstddef>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "rocketphys/Landing.h"
#include "rocketphys/Thruster.h"

namespace rocketphys {

    class RigidBody {
    public:
        // ---------- Setup ----------
        // A mass <= 0 makes the body static (invMass_ = 0, it never moves).
        void SetMass(float mass);

        // Call AFTER SetMass(): inertia depends on mass.
        void SetBoxInertia(float width, float height, float depth);

        // Call AFTER SetMass(). Long axis is Y. Also sets the half-height and radius.
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

        // ---------- Accessors ----------
        const glm::vec3& GetPosition() const { return position_; }
        const glm::vec3& GetVelocity() const { return velocity_; }
        const glm::quat& GetOrientation() const { return orientation_; }
        const glm::vec3& GetAngularVelocity() const { return angularVelocity_; }
        float GetMass() const { return mass_; }
        float GetInvMass() const { return invMass_; }
        float GetHalfHeight() const { return halfHeight_; }
        glm::mat3 GetRotationMatrix() const { return glm::mat3_cast(orientation_); }

        // Lowest world-space point of the cylinder: the contact point for the floor.
        glm::vec3 GetLowestPoint() const;

        // Angle between the body's up axis and world up, in degrees. 0 = upright.
        float GetTiltDegrees() const;

        void SetPosition(const glm::vec3& p) { position_ = p; }
        void SetVelocity(const glm::vec3& v) { velocity_ = v; }
        void SetOrientation(const glm::quat& q) { orientation_ = glm::normalize(q); }
        void SetAngularVelocity(const glm::vec3& w) { angularVelocity_ = w; }

        // Zero velocities and accumulators (position/orientation are set separately).
        void ZeroMotion();

        // ---------- Landing ----------
        LandingStatus GetLandingStatus() const { return landingStatus_; }
        float GetTouchdownSpeed() const { return touchdownSpeed_; }
        float GetTouchdownTiltDeg() const { return touchdownTiltDeg_; }

        // Called by World on the first ground contact. Records speed and tilt, and
        // classifies the landing. Does nothing after the first call until ResetLandingStatus().
        void RecordTouchdown(float speedMps, float tiltDeg);

        // Sets status back to Flying and clears the touchdown values.
        // Kept separate from ZeroMotion() so a landing result isn't erased by zeroing velocity.
        void ResetLandingStatus();

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

        // Shape and tuning
        float linearDamping_ = 1.0f;
        float angularDamping_ = 1.0f;
        float halfHeight_ = 0.5f;
        float radius_ = 0.0f;  // cylinder radius, m. 0 means a vertical line segment.

        // Landing record
        LandingStatus landingStatus_ = LandingStatus::Flying;
        float touchdownSpeed_ = 0.0f;
        float touchdownTiltDeg_ = 0.0f;

        std::vector<Thruster> thrusters_;
    };

}  // namespace rocketphys