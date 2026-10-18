#include "rocketphys/LandingEnv.h"

#include <algorithm>
#include <cmath>

#include "rocketphys/Rocket.h"

namespace rocketphys {

    namespace {

        constexpr float kTwoPi = 6.28318531f;
        constexpr float kHalfHeight = 4.0f;     // MakeRocketBody() half-height, m
        constexpr float kDescentZone = 30.0f;   // m, descent penalty applies below this altitude
        constexpr float kObsLimit = 10.0f;      // observation clamp

        float Clamp(float x, float lo, float hi) { return std::clamp(x, lo, hi); }

    }  // namespace

    LandingEnv::LandingEnv(const EnvConfig& config) : config_(config) {
        config_.difficulty = Clamp(config_.difficulty, 0.0f, 1.0f);
        config_.actionRepeat = std::max(1, config_.actionRepeat);
        Reset(0);
    }

    void LandingEnv::BuildWorld() {
        world_ = World();
        world_.SetGravity({ 0.0f, -9.81f, 0.0f });
        rocketId_ = world_.AddBody(MakeRocketBody());
        done_ = false;
    }

    void LandingEnv::SetDifficulty(float d) {
        config_.difficulty = Clamp(d, 0.0f, 1.0f);
    }

    Observation LandingEnv::Reset(std::uint32_t seed) {
        rng_.seed(seed);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        const float d = config_.difficulty;

        // Height: uniform over the configured range. Not scaled by difficulty.
        const float u0 = U(rng_);
        const float height = config_.minStartHeight +
            (config_.maxStartHeight - config_.minStartHeight) * u0;

        // Orientation: uniform random rotation (Shoemake's method), blended from upright by difficulty.
        // Parameter names: u1 picks the magnitude split, u2 and u3 pick the two angles.
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
        // Each random draw is its own statement so the order is fixed on every compiler.
        const float vmax = config_.maxStartSpeed * d;
        const float u4 = U(rng_);
        const float u5 = U(rng_);
        const float u6 = U(rng_);
        const glm::vec3 velocity((2.0f * u4 - 1.0f) * vmax, -u5 * vmax, (2.0f * u6 - 1.0f) * vmax);

        // Position: the body center sits kHalfHeight above the requested lowest-point altitude.
        const glm::vec3 position(0.0f, height + kHalfHeight, 0.0f);

        return ResetFromState(position, orientation, velocity, glm::vec3(0.0f));
    }

    Observation LandingEnv::ResetFromState(const glm::vec3& position, const glm::quat& orientation,
        const glm::vec3& velocity, const glm::vec3& angularVelocity) {
        BuildWorld();
        RigidBody& rocket = world_.GetBody(rocketId_);
        rocket.SetPosition(position);
        rocket.SetOrientation(orientation);
        rocket.SetVelocity(velocity);
        rocket.SetAngularVelocity(angularVelocity);
        return GetObservation();
    }

    Observation LandingEnv::GetObservation() const {
        const RigidBody& r = Rocket();
        const glm::vec3 p = r.GetPosition();
        const glm::vec3 v = r.GetVelocity();
        const glm::vec3 w = r.GetAngularVelocity();
        const glm::mat3 R = r.GetRotationMatrix();

        Observation o{};
        std::size_t k = 0;
        o[k++] = r.GetLowestPoint().y / 100.0f;
        o[k++] = v.x / 50.0f;
        o[k++] = v.y / 50.0f;
        o[k++] = v.z / 50.0f;
        o[k++] = p.x / 100.0f;
        o[k++] = p.z / 100.0f;
        for (int c = 0; c < 3; ++c) {
            for (int row = 0; row < 3; ++row) {
                o[k++] = R[c][row];
            }
        }
        o[k++] = w.x / 10.0f;
        o[k++] = w.y / 10.0f;
        o[k++] = w.z / 10.0f;

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

        float reward = -w.stepCost - w.tiltCost * (tilt / 90.0f) - w.spinCost * spin;
        if (altitude < kDescentZone) {
            reward -= w.descentCost * (fallSpeed / 10.0f);
        }
        return reward;
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

        // After the episode ends, further steps do nothing and give no reward.
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
        result.reward = result.terminated ? TerminalReward() : ShapingReward();
        result.observation = GetObservation();

        done_ = result.terminated || result.truncated;
        return result;
    }

}  // namespace rocketphys