#include "rocketphys/Baseline.h"

#include <algorithm>
#include <cmath>

namespace rocketphys {

    namespace {

        constexpr float kGravity = 9.81f;
        constexpr float kPairTorque = 6000.0f;  // one diagonal pair at full throttle, N*m

        float Clamp(float x, float lo, float hi) { return std::clamp(x, lo, hi); }

    }  // namespace

    Action LandingBaseline::Compute(const RigidBody& rocket) const {
        const glm::vec3 p = rocket.GetPosition();
        const glm::vec3 v = rocket.GetVelocity();
        const glm::vec3 w = rocket.GetAngularVelocity();
        const glm::mat3 R = rocket.GetRotationMatrix();
        const glm::vec3 bodyUp = R * glm::vec3(0.0f, 1.0f, 0.0f);

        // 0. Blend t: 0 = landing control, 1 = flip control. Continuous in the tilt.
        const float cosLand = std::cos(glm::radians(gains_.landTiltDeg));
        const float cosRecover = std::cos(glm::radians(gains_.recoverTiltDeg));
        const float t = Clamp((cosLand - bodyUp.y) / (cosLand - cosRecover), 0.0f, 1.0f);

        // 1. Lateral control pulls toward the pad. It fades out during a flip, because
        //    leaning while inverted only adds error.
        const float lateralScale = 1.0f - t;
        float ax = -(gains_.lateralPosGain * p.x + gains_.lateralVelGain * v.x) * lateralScale;
        float az = -(gains_.lateralPosGain * p.z + gains_.lateralVelGain * v.z) * lateralScale;

        // Cap the lean so the horizontal acceleration never exceeds g * tan(maxTilt).
        const float aMax = kGravity * std::tan(glm::radians(gains_.maxTiltDeg));
        const float aLen = std::sqrt(ax * ax + az * az);
        if (aLen > aMax) {
            ax *= aMax / aLen;
            az *= aMax / aLen;
        }

        // 2. Desired up direction.
        const glm::vec3 upDesired = glm::normalize(glm::vec3(ax / kGravity, 1.0f, az / kGravity));

        // 3. Attitude error: the axis that turns the body up toward upDesired.
        glm::vec3 errorWorld = glm::cross(bodyUp, upDesired);
        if (bodyUp.y < 0.0f && glm::length(errorWorld) < 0.5f) {
            // Past 90 degrees the error shrinks toward zero at 180 degrees, so the controller
            // would stall upside down. Give it a fixed direction and a usable size.
            glm::vec3 dir = errorWorld;
            if (glm::length(dir) < 1e-4f) dir = glm::cross(bodyUp, glm::vec3(0.0f, 0.0f, 1.0f));
            if (glm::length(dir) < 1e-4f) dir = glm::cross(bodyUp, glm::vec3(1.0f, 0.0f, 0.0f));
            errorWorld = glm::normalize(dir) * 0.5f;
        }

        // 4. PD torque, with gains blended between landing and flip.
        const float kp = (1.0f - t) * gains_.attitude.kp + t * gains_.flipKp;
        const float kd = (1.0f - t) * gains_.attitude.kd + t * gains_.flipKd;
        const glm::vec3 torqueWorld = kp * errorWorld - kd * w;
        const glm::vec3 torqueBody = glm::transpose(R) * torqueWorld;

        // Positive X torque fires pitch pair B (positive pitch action).
        // Positive Z torque fires roll pair A (positive roll action).
        const float pitch = Clamp(torqueBody.x / kPairTorque, -1.0f, 1.0f);
        const float roll = Clamp(torqueBody.z / kPairTorque, -1.0f, 1.0f);

        // 5. Vertical control. Thrust fades out as the rocket flips, because the main engine
        //    would push the body into the ground while inverted.
        const float altitude = rocket.GetLowestPoint().y;
        const float vTarget = -Clamp(gains_.altitudeGain * altitude, gains_.minDescent, gains_.maxDescent);
        const float aVert = gains_.verticalGain * (vTarget - v.y);

        // Thrust along the body axis has vertical part T * bodyUp.y. The divisor is clamped
        // at 0.5 (60 degrees) so the demand stays finite. Above landTiltDeg, t > 0 and thrust
        // is blended out, so the clamp is never what limits the landing.
        const float cosLean = Clamp(bodyUp.y, 0.5f, 1.0f);
        const float thrustWanted = rocket.GetMass() * (kGravity + aVert) / cosLean;
        const float thrustFactor = 1.0f - t;

        const float maxThrust = rocket.GetThrusters()[kMainEngine].maxForce;
        const float throttle = Clamp(thrustFactor * thrustWanted / maxThrust, 0.0f, 1.0f);

        // The environment maps throttle in [0, 1] to the action in [-1, 1] as (a + 1) / 2,
        // so the inverse is a = 2 * throttle - 1.
        return Action{ 2.0f * throttle - 1.0f, pitch, roll };
    }

}  // namespace rocketphys