#pragma once
// LandingBaseline: a hand-written landing controller. It reads the rocket's state and
// returns an Action in the same format the RL policy uses, so both run through LandingEnv.

#include "rocketphys/LandingEnv.h"  // Action
#include "rocketphys/RigidBody.h"
#include "rocketphys/Rocket.h"      // AttitudeGains, kMainEngine

namespace rocketphys {

    struct BaselineGains {
        // Descent: target vertical speed = -clamp(altitudeGain * h, minDescent, maxDescent).
        float altitudeGain = 0.3f;      // 1/s
        float minDescent = 0.6f;        // m/s, slowest approach near the ground
        float maxDescent = 8.0f;        // m/s, fastest descent from high up

        // Vertical loop: commanded acceleration = verticalGain * (target speed - vy).
        float verticalGain = 0.8f;      // 1/s

        // Lateral: pulls toward the pad and damps sideways speed.
        float lateralPosGain = 0.05f;   // 1/s^2
        float lateralVelGain = 0.5f;    // 1/s
        float maxTiltDeg = 15.0f;       // largest lean used to move sideways

        // Attitude in landing mode, used below landTiltDeg. Same PD gains as UpdateRocketAttitude.
        AttitudeGains attitude;

        // Blend between landing and flip control as the tilt goes from landTiltDeg to recoverTiltDeg.
        // The blend depends only on the current tilt, so there is no state to reset between episodes.
        float landTiltDeg = 30.0f;
        float recoverTiltDeg = 60.0f;

        // Flip-mode attitude gains. With kp = 6000, a 1 rad (57 degree) error saturates one
        // RCS pair, so the flip uses full authority. kd gives a damping ratio of about 0.5.
        float flipKp = 6000.0f;         // N*m per radian
        float flipKd = 6000.0f;         // N*m*s per radian/s
    };

    class LandingBaseline {
    public:
        explicit LandingBaseline(const BaselineGains& gains = BaselineGains{}) : gains_(gains) {}

        // Returns [throttle, pitch, roll], each in [-1, 1], for the rocket's current state.
        Action Compute(const RigidBody& rocket) const;

        const BaselineGains& Gains() const { return gains_; }

    private:
        BaselineGains gains_;
    };

}  // namespace rocketphys