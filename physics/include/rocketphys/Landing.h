#pragma once
// Landing rules. Pure data and one classification function, no physics.

namespace rocketphys {

    enum class LandingStatus {
        Flying,
        Landed,
        Crashed
    };

    // Soft-landing limits. Tune these, not the logic.
    inline constexpr float kSoftLandingSpeed = 2.0f;     // m/s, maximum touchdown speed
    inline constexpr float kSoftLandingTiltDeg = 10.0f;  // degrees, maximum touchdown tilt

    // Returns Landed only if both speed and tilt are inside the limits.
    inline LandingStatus ClassifyTouchdown(float speedMps, float tiltDeg) {
        return (speedMps < kSoftLandingSpeed && tiltDeg < kSoftLandingTiltDeg)
            ? LandingStatus::Landed
            : LandingStatus::Crashed;
    }

}  // namespace rocketphys