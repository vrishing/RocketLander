#include "rocketphys/Rocket.h"

#include <cmath>

#include "rocketphys/Thruster.h"

namespace rocketphys {

    namespace {

        // One diagonal pair at full throttle: two thrusters at 3000 N*m each.
        constexpr float kPairTorque = 6000.0f;

        float Clamp01(float x) {
            return (x < 0.0f) ? 0.0f : (x > 1.0f) ? 1.0f : x;
        }

    }  // namespace

    RigidBody MakeRocketBody() {
        RigidBody rocket;
        rocket.SetMass(1000.0f);
        rocket.SetCylinderInertia(0.5f, 8.0f);  // radius, height. Must come after SetMass.
        rocket.SetDamping(1.0f, 1.0f);          // no air in the simulation
        rocket.SetHalfHeight(4.0f);             // center to lowest point

        // Order here MUST match the k... index constants in Rocket.h.
        rocket.AddThruster(Thruster({ 0.0f, -4.0f,  0.0f }, { 0.0f, 1.0f,  0.0f }, 25000.0f));  // main

        rocket.AddThruster(Thruster({ 0.0f,  3.0f,  0.5f }, { 0.0f, 0.0f, -1.0f }, 1000.0f));  // pitch A top
        rocket.AddThruster(Thruster({ 0.0f, -3.0f, -0.5f }, { 0.0f, 0.0f,  1.0f }, 1000.0f));  // pitch A bottom
        rocket.AddThruster(Thruster({ 0.0f,  3.0f, -0.5f }, { 0.0f, 0.0f,  1.0f }, 1000.0f));  // pitch B top
        rocket.AddThruster(Thruster({ 0.0f, -3.0f,  0.5f }, { 0.0f, 0.0f, -1.0f }, 1000.0f));  // pitch B bottom

        rocket.AddThruster(Thruster({ 0.5f,  3.0f,  0.0f }, { -1.0f, 0.0f,  0.0f }, 1000.0f));  // roll A top
        rocket.AddThruster(Thruster({ -0.5f, -3.0f,  0.0f }, { 1.0f, 0.0f,  0.0f }, 1000.0f));  // roll A bottom
        rocket.AddThruster(Thruster({ -0.5f,  3.0f,  0.0f }, { 1.0f, 0.0f,  0.0f }, 1000.0f));  // roll B top
        rocket.AddThruster(Thruster({ 0.5f, -3.0f,  0.0f }, { -1.0f, 0.0f,  0.0f }, 1000.0f));  // roll B bottom

        return rocket;
    }

    glm::vec3 RocketTiltError(const RigidBody& rocket) {
        const glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
        const glm::vec3 bodyUp = rocket.GetRotationMatrix() * glm::vec3(0.0f, 1.0f, 0.0f);
        return glm::cross(bodyUp, worldUp);
    }

    void UpdateRocketAttitude(RigidBody& rocket, const AttitudeGains& gains) {
        // 1. Desired torque in world space: PD on tilt, damping on spin rate.
        const glm::vec3 torqueWorld =
            gains.kp * RocketTiltError(rocket) - gains.kd * rocket.GetAngularVelocity();

        // 2. Thrusters are defined in body axes, so convert the torque into body axes.
        const glm::vec3 torqueBody = glm::transpose(rocket.GetRotationMatrix()) * torqueWorld;

        // 3. Pitch (X torque). Negative fires pair A, positive fires pair B.
        const float pitch = Clamp01(std::fabs(torqueBody.x) / kPairTorque);
        const float pitchA = (torqueBody.x < 0.0f) ? pitch : 0.0f;
        const float pitchB = (torqueBody.x > 0.0f) ? pitch : 0.0f;
        rocket.SetThrottle(kPitchATop, pitchA);
        rocket.SetThrottle(kPitchABottom, pitchA);
        rocket.SetThrottle(kPitchBTop, pitchB);
        rocket.SetThrottle(kPitchBBottom, pitchB);

        // 4. Roll (Z torque). Positive fires pair A, negative fires pair B.
        const float roll = Clamp01(std::fabs(torqueBody.z) / kPairTorque);
        const float rollA = (torqueBody.z > 0.0f) ? roll : 0.0f;
        const float rollB = (torqueBody.z < 0.0f) ? roll : 0.0f;
        rocket.SetThrottle(kRollATop, rollA);
        rocket.SetThrottle(kRollABottom, rollA);
        rocket.SetThrottle(kRollBTop, rollB);
        rocket.SetThrottle(kRollBBottom, rollB);
    }

}  // namespace rocketphys