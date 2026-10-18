#include "rocketphys/LandingEnv.h"

#include <algorithm>
#include <cmath>

#include "rocketphys/Rocket.h"

namespace rocketphys {

    namespace {

        constexpr float kTwoPi = 6.28318531f;
        // constexpr float kDescentZone = 30.0f;   // m, descent penalty applies below this altitude
        constexpr float kObsLimit = 10.0f;      // observation clamp

        float Clamp(float x, float lo, float hi) { return std::clamp(x, lo, hi); }
        // Fastest descent that is still safe at this altitude: v = sqrt(2 a h), never below the minimum.
        float SafeDescentSpeed(float altitude, const RewardWeights& w) {
            return std::max(w.minSafeSpeed, std::sqrt(2.0f * w.brakingDecel * std::max(altitude, 0.0f)));
        }
    }  // namespace

    LandingEnv::LandingEnv(const EnvConfig& config) : config_(config) {
        config_.difficulty = Clamp(config_.difficulty, 0.0f, 1.0f);
        config_.actionRepeat = std::max(1, config_.actionRepeat);
        config_.maxWindSpeed = std::max(0.0f, config_.maxWindSpeed);
        config_.gustInterval = std::max(0.1f, config_.gustInterval);
        Reset(0);
    }

    void LandingEnv::BuildWorld(std::uint32_t seed) {
        world_ = World();
        world_.SetGravity({ 0.0f, -9.81f, 0.0f });

        // Air and wind are both optional. Wind needs the atmosphere, because it acts through drag.
        if (config_.atmosphere) {
            world_.SetAtmosphere(true);
            world_.SetBaseWind(config_.baseWind);
        }
        if (config_.atmosphere && config_.maxWindSpeed > 0.0f) {
            // Gusts are seeded from the episode seed, so the same seed gives the same wind history.
            world_.SetGusts(true, config_.maxWindSpeed, config_.gustInterval, seed);
        }

        rocketId_ = world_.AddBody(MakeRocketBody());
        done_ = false;
    }

    void LandingEnv::SetDifficulty(float d) {
        config_.difficulty = Clamp(d, 0.0f, 1.0f);
    }

    std::size_t LandingEnv::ObservationSize() const {
        return config_.windSensor ? kBaseObsSize + kWindSensorObsSize : kBaseObsSize;
    }

    Observation LandingEnv::Reset(std::uint32_t seed) {
        rng_.seed(seed);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float d = config_.difficulty;

        // Altitude of the LOWEST point, uniform over the configured range.
        const float u0 = U(rng_);
        const float altitude = config_.minStartHeight +
            (config_.maxStartHeight - config_.minStartHeight) * u0;

        // Orientation: uniform random rotation (Shoemake's method), blended from upright by difficulty.
        const float u1 = U(rng_);
        const float u2 = U(rng_);
        const float u3 = U(rng_);
        const float s1 = std::sqrt(1.0f - u1);
        const float s2 = std::sqrt(u1);
        const glm::quat randomQ = glm::normalize(glm::quat(
            s2 * std::cos(kTwoPi * u3),   // w
            s1 * std::sin(kTwoPi * u2),   // x
            s1 * std::cos(kTwoPi * u2),   // y
            s2 * std::sin(kTwoPi * u3))); // z
        const glm::quat orientation =
            glm::normalize(glm::slerp(glm::quat(1.0f, 0.0f, 0.0f, 0.0f), randomQ, d));

        // Velocity: sideways in both directions, downward only, scaled by difficulty.
        const float vmax = config_.maxStartSpeed * d;
        const float u4 = U(rng_);
        const float u5 = U(rng_);
        const float u6 = U(rng_);
        const glm::vec3 velocity((2.0f * u4 - 1.0f) * vmax, -u5 * vmax, (2.0f * u6 - 1.0f) * vmax);

        // The wind is seeded separately from the start, so the start and the wind are independent.
        BuildWorld(seed);
        return ResetFromLowestPoint(altitude, 0.0f, 0.0f, orientation, velocity, glm::vec3(0.0f));
    }

    Observation LandingEnv::ResetFromState(const glm::vec3& position, const glm::quat& orientation,
        const glm::vec3& velocity, const glm::vec3& angularVelocity) {
        BuildWorld(0);
        RigidBody& rocket = world_.GetBody(rocketId_);
        rocket.SetPosition(position);
        rocket.SetOrientation(orientation);
        rocket.SetVelocity(velocity);
        rocket.SetAngularVelocity(angularVelocity);
        prevPotential_ = Potential();
        return GetObservation();
    }

    Observation LandingEnv::ResetFromLowestPoint(float lowestAltitude, float x, float z,
        const glm::quat& orientation,
        const glm::vec3& velocity,
        const glm::vec3& angularVelocity) {
        RigidBody probe = MakeRocketBody();
        probe.SetPosition(glm::vec3(0.0f));
        probe.SetOrientation(orientation);
        const float offsetY = probe.GetLowestPoint().y;

        const glm::vec3 center(x, lowestAltitude - offsetY, z);
        return ResetFromState(center, orientation, velocity, angularVelocity);
    }

    Observation LandingEnv::GetObservation() const {
        const RigidBody& r = Rocket();
        const glm::vec3 p = r.GetPosition();
        const glm::vec3 v = r.GetVelocity();
        const glm::vec3 w = r.GetAngularVelocity();
        const glm::mat3 R = r.GetRotationMatrix();

        Observation o;
        o.reserve(ObservationSize());
        o.push_back(r.GetLowestPoint().y / 100.0f);
        o.push_back(v.x / 50.0f);
        o.push_back(v.y / 50.0f);
        o.push_back(v.z / 50.0f);
        o.push_back(p.x / 100.0f);
        o.push_back(p.z / 100.0f);
        for (int c = 0; c < 3; ++c) {
            for (int row = 0; row < 3; ++row) {
                o.push_back(R[c][row]);
            }
        }
        o.push_back(w.x / 10.0f);
        o.push_back(w.y / 10.0f);
        o.push_back(w.z / 10.0f);

        if (config_.windSensor) {
            // Air-relative velocity: what a pitot tube would read. Zero wind gives the body velocity.
            const glm::vec3 relV = v - world_.GetWindVelocity();
            o.push_back(relV.x / 50.0f);
            o.push_back(relV.y / 50.0f);
            o.push_back(relV.z / 50.0f);
        }

        for (float& value : o) {
            value = Clamp(value, -kObsLimit, kObsLimit);
        }
        return o;
    }

    void LandingEnv::ApplyAction(const Action& action) {
        RigidBody& r = world_.GetBody(rocketId_);

        const float throttle = Clamp((action[0] + 1.0f) * 0.5f, 0.0f, 1.0f);
        const float pitch = Clamp(action[1], -1.0f, 1.0f);
        const float roll = Clamp(action[2], -1.0f, 1.0f);

        r.SetThrottle(kMainEngine, throttle);

        // Pitch: positive fires pair B, negative fires pair A.
        const float pitchB = std::max(pitch, 0.0f);
        const float pitchA = std::max(-pitch, 0.0f);
        r.SetThrottle(kPitchBTop, pitchB);
        r.SetThrottle(kPitchBBottom, pitchB);
        r.SetThrottle(kPitchATop, pitchA);
        r.SetThrottle(kPitchABottom, pitchA);

        // Roll: positive fires pair A, negative fires pair B.
        const float rollA = std::max(roll, 0.0f);
        const float rollB = std::max(-roll, 0.0f);
        r.SetThrottle(kRollATop, rollA);
        r.SetThrottle(kRollABottom, rollA);
        r.SetThrottle(kRollBTop, rollB);
        r.SetThrottle(kRollBBottom, rollB);
    }

    float LandingEnv::ShapingReward() const {
        const RigidBody& r = Rocket();
        const RewardWeights& w = config_.reward;

        const float tilt = r.GetTiltDegrees();
        const float spin = glm::length(r.GetAngularVelocity());
        const float altitude = r.GetLowestPoint().y;
        const float fallSpeed = std::max(0.0f, -r.GetVelocity().y);

        // Penalty only for the part of the fall speed above the envelope. The cap of 30 m/s
        // stops a single violent step from swamping the other terms.
        const float excess = std::max(0.0f, fallSpeed - SafeDescentSpeed(altitude, w));
        const float envelope = w.speedEnvelopeCost * std::min(excess, 30.0f);

        return -w.stepCost - w.tiltCost * (tilt / 90.0f) - w.spinCost * spin - envelope;
    }

    float LandingEnv::Potential() const {
        // Terminal states have potential zero. This keeps the telescoping sum exact.
        if (Rocket().GetLandingStatus() != LandingStatus::Flying) return 0.0f;
        return -config_.reward.altitudeShaping * Rocket().GetLowestPoint().y;
    }

    float LandingEnv::TerminalReward() const {
        const RigidBody& r = Rocket();
        const RewardWeights& w = config_.reward;

        if (r.GetLandingStatus() == LandingStatus::Landed) {
            return w.landedBonus
                - w.touchdownSpeedCost * r.GetTouchdownSpeed()
                - w.touchdownTiltCost * r.GetTouchdownTiltDeg();
        }
        return w.crashPenalty;
    }

    StepResult LandingEnv::Step(const Action& action) {
        StepResult result;

        if (done_) {
            result.observation = GetObservation();
            result.terminated = true;
            result.status = Rocket().GetLandingStatus();
            return result;
        }

        ApplyAction(action);

        for (int i = 0; i < config_.actionRepeat; ++i) {
            world_.Step();
            if (Rocket().GetLandingStatus() != LandingStatus::Flying) break;
        }

        result.status = Rocket().GetLandingStatus();
        result.terminated = (result.status != LandingStatus::Flying);
        result.truncated = !result.terminated && SimTime() >= config_.maxEpisodeSeconds;

        // Potential-based shaping: gamma * Phi(next state) - Phi(this state).
        // Summed over an episode it reduces to the start and end values only, so it rewards
        // descent without changing which policy is best.
        const float phiNow = Potential();
        float reward = ShapingReward() + config_.reward.discount * phiNow - prevPotential_;
        prevPotential_ = phiNow;

        if (result.terminated) reward += TerminalReward();
        if (result.truncated) reward += config_.reward.timeoutPenalty;

        result.reward = reward;
        result.observation = GetObservation();
        done_ = result.terminated || result.truncated;
        return result;
    }

}  // namespace rocketphys