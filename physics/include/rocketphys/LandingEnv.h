#pragma once
// LandingEnv: one headless rocket-landing episode for reinforcement learning.
// Pure C++, no rendering. The Python binding and Gymnasium wrapper sit on top of this class.

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "rocketphys/Landing.h"
#include "rocketphys/RigidBody.h"
#include "rocketphys/World.h"

namespace rocketphys {

    // Action: [throttle, pitch, roll], each in [-1, 1].
    //   throttle: main engine. -1 = off, 0 = half power, about -0.215 = hover, +1 = full.
    //   pitch:    +1 = positive X torque (fires pitch pair B), -1 = negative X torque (pair A).
    //   roll:     +1 = positive Z torque (fires roll pair A),  -1 = negative Z torque (pair B).
    using Action = std::array<float, 3>;

    // Observation layout. The base layout is 18 values. With the wind sensor on, 3 more are added.
    //   [0]      altitude of the lowest point / 100
    //   [1..3]   velocity / 50
    //   [4..5]   horizontal position x, z / 100
    //   [6..14]  rotation matrix, column-major, 9 values
    //   [15..17] angular velocity / 10
    //   [18..20] air-relative velocity / 50 (only when the wind sensor is on)
    // Every value is clamped to [-10, 10].
    inline constexpr std::size_t kBaseObsSize = 18;
    inline constexpr std::size_t kWindSensorObsSize = 3;
    using Observation = std::vector<float>;

    struct RewardWeights {
        float discount = 0.999f;            // must match the PPO gamma in train.py
        float stepCost = 0.01f;             // per policy step
        float tiltCost = 0.05f;             // per policy step, times tilt / 90 deg
        float spinCost = 0.02f;             // per policy step, times |omega| in rad/s
        float altitudeShaping = 0.05f;      // potential: reward per metre of height lost
        float landedBonus = 100.0f;         // terminal, on Landed
        float crashPenalty = -100.0f;       // terminal, on Crashed
        float timeoutPenalty = -50.0f;      // terminal, when time runs out while still flying
        float touchdownSpeedCost = 5.0f;    // terminal, times touchdown speed in m/s
        float touchdownTiltCost = 0.2f;     // terminal, times touchdown tilt in degrees
    };

    struct EnvConfig {
        int actionRepeat = 4;               // physics steps per policy step (4 at 120 Hz = 30 Hz control)
        float maxEpisodeSeconds = 60.0f;    // simulated-time limit, gives a truncation
        float minStartHeight = 15.0f;       // m, altitude of the lowest point at the start
        float maxStartHeight = 200.0f;      // m
        float maxStartSpeed = 5.0f;         // m/s, at difficulty 1
        float difficulty = 1.0f;            // 0 = upright and still, 1 = full random start

        // Air. Off by default, so existing behavior is unchanged.
        bool atmosphere = false;
        float maxWindSpeed = 0.0f;          // m/s, peak gust speed. 0 means no wind.
        glm::vec3 baseWind{ 0.0f };           // m/s, steady wind added to any gusts. Needs atmosphere.
        float gustInterval = 4.0f;          // s between gust targets
        bool windSensor = false;            // adds the air-relative velocity to the observation

        RewardWeights reward;
    };

    struct StepResult {
        Observation observation;
        float reward = 0.0f;
        bool terminated = false;            // landed or crashed
        bool truncated = false;             // time limit reached
        LandingStatus status = LandingStatus::Flying;
    };

    class LandingEnv {
    public:
        explicit LandingEnv(const EnvConfig& config = EnvConfig{});

        // Random start from the seed. Same seed gives the same start and the same wind, bit for bit.
        Observation Reset(std::uint32_t seed);

        // Explicit start, for tests, curricula, and baselines.
        Observation ResetFromState(const glm::vec3& position, const glm::quat& orientation,
            const glm::vec3& velocity, const glm::vec3& angularVelocity);

        // Start with the LOWEST point at lowestAltitude (m), for any tilt.
        Observation ResetFromLowestPoint(float lowestAltitude, float x, float z,
            const glm::quat& orientation,
            const glm::vec3& velocity,
            const glm::vec3& angularVelocity);

        // Applies the action, holds it for actionRepeat physics steps, and returns the result.
        StepResult Step(const Action& action);

        Observation GetObservation() const;

        // Observation size for the current config: 18, or 21 with the wind sensor.
        std::size_t ObservationSize() const;

        const RigidBody& Rocket() const { return world_.GetBody(rocketId_); }
        float SimTime() const { return static_cast<float>(world_.GetTime()); }

        void SetDifficulty(float d);
        const EnvConfig& Config() const { return config_; }

    private:
        void BuildWorld(std::uint32_t seed);
        void ApplyAction(const Action& action);
        float ShapingReward() const;
        float TerminalReward() const;

        EnvConfig config_;
        World world_;
        std::size_t rocketId_ = 0;
        bool done_ = false;
        std::mt19937 rng_{ 0 };
        float Potential() const;            // altitude potential of the current state
        float prevPotential_ = 0.0f;        // potential at the previous policy step
    };

}  // namespace rocketphys