#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "rocketphys/LandingEnv.h"
#include <cstdio>

#include "rocketphys/Baseline.h"

using namespace rocketphys;

namespace {

    const Action kCoast{ -1.0f, 0.0f, 0.0f };  // no main thrust, no RCS

    // Steps with the same action until the episode ends, or the step limit is reached.
    StepResult RunUntilDone(LandingEnv& env, const Action& action, int maxPolicySteps) {
        StepResult r;
        for (int i = 0; i < maxPolicySteps; ++i) {
            r = env.Step(action);
            if (r.terminated || r.truncated) break;
        }
        return r;
    }

}  // namespace

// Same seed gives the same starting state. Different seeds give different starts.
TEST(Env, SameSeedGivesSameStart) {
    LandingEnv a, b;
    const Observation oa = a.Reset(42);
    const Observation ob = b.Reset(42);
    const Observation oc = b.Reset(43);
    EXPECT_EQ(oa, ob);
    EXPECT_NE(oa, oc);
}

// Random actions must never produce NaN or values outside the clamp.
TEST(Env, ObservationStaysFiniteAndBounded) {
    LandingEnv env;
    env.Reset(7);

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    for (int i = 0; i < 300; ++i) {
        const Action a{ U(rng), U(rng), U(rng) };
        const StepResult r = env.Step(a);
        for (float v : r.observation) {
            ASSERT_TRUE(std::isfinite(v));
            ASSERT_LE(std::fabs(v), 10.0f + 1e-4f);
        }
        if (r.terminated || r.truncated) break;
    }
}

// Every episode must end, either by landing or crashing, or by the time limit.
TEST(Env, RandomEpisodeAlwaysEnds) {
    LandingEnv env;
    env.Reset(11);

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);

    bool ended = false;
    for (int i = 0; i < 2000; ++i) {
        const StepResult r = env.Step({ U(rng), U(rng), U(rng) });
        if (r.terminated || r.truncated) {
            ended = true;
            break;
        }
    }
    EXPECT_TRUE(ended);
}

// Dropped from 100 m with no thrust: impact speed is about 44 m/s, so it must crash.
TEST(Env, FreeFallFromHeightCrashes) {
    LandingEnv env;
    env.ResetFromState({ 0.0f, 104.0f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));

    const StepResult r = RunUntilDone(env, kCoast, 2000);

    EXPECT_TRUE(r.terminated);
    EXPECT_EQ(r.status, LandingStatus::Crashed);
    EXPECT_GT(env.Rocket().GetTouchdownSpeed(), 40.0f);
    EXPECT_LT(r.reward, -50.0f);
}

// A gentle upright landing (5 cm drop, about 1 m/s) earns the landed bonus.
TEST(Env, SoftLandingEarnsBonus) {
    LandingEnv env;
    env.ResetFromState({ 0.0f, 4.05f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));

    const StepResult r = RunUntilDone(env, kCoast, 2000);

    EXPECT_TRUE(r.terminated);
    EXPECT_EQ(r.status, LandingStatus::Landed);
    EXPECT_GT(r.reward, 90.0f);
}

// Stepping after the episode ends must not give more reward.
TEST(Env, StepAfterEndGivesNoReward) {
    LandingEnv env;
    env.ResetFromState({ 0.0f, 4.05f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));
    RunUntilDone(env, kCoast, 2000);

    const StepResult after = env.Step(kCoast);
    EXPECT_EQ(after.reward, 0.0f);
    EXPECT_TRUE(after.terminated);
}

// Hover action should keep the rocket near its start height for 2 s.
TEST(Env, HoverActionHoldsHeight) {
    LandingEnv env;
    env.ResetFromState({ 0.0f, 104.0f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));

    // Throttle = 9810 / 25000 = 0.3924, so the action is 2 * 0.3924 - 1 = -0.2152.
    const Action hover{ -0.2152f, 0.0f, 0.0f };
    StepResult r;
    for (int i = 0; i < 60; ++i) r = env.Step(hover);  // 60 policy steps = 2 s

    EXPECT_FALSE(r.terminated);
    EXPECT_NEAR(env.Rocket().GetLowestPoint().y, 100.0f, 0.5f);
}

// A 30 degree tilted start with its lowest point at 50 m must report 0.5 (50 / 100) in the
// altitude slot, and the physics must agree. The old code got this wrong for any tilt.
TEST(Env, TiltedStartPlacesLowestPointAtRequestedAltitude) {
    LandingEnv env;
    const glm::quat tilt = glm::angleAxis(glm::radians(30.0f), glm::vec3(1.0f, 0.0f, 0.0f));

    const Observation o = env.ResetFromLowestPoint(50.0f, 0.0f, 0.0f, tilt,
        glm::vec3(0.0f), glm::vec3(0.0f));

    EXPECT_NEAR(o[0], 0.5f, 1e-4f);
    EXPECT_NEAR(env.Rocket().GetLowestPoint().y, 50.0f, 1e-4f);
}

// An upright start must match the old behavior: the body center sits half the height above
// the lowest point, which is 4 m for this rocket.
TEST(Env, UprightStartCenterIsFourMetresAboveLowestPoint) {
    LandingEnv env;
    env.ResetFromLowestPoint(20.0f, 0.0f, 0.0f, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));

    EXPECT_NEAR(env.Rocket().GetPosition().y, 24.0f, 1e-4f);
}

// Every random start must place the lowest point inside the configured altitude range,
// for any tilt that difficulty allows.
TEST(Env, RandomStartLowestPointIsInConfiguredRange) {
    EnvConfig cfg;
    cfg.minStartHeight = 15.0f;
    cfg.maxStartHeight = 200.0f;
    cfg.difficulty = 1.0f;
    LandingEnv env(cfg);

    for (std::uint32_t seed = 1; seed <= 50; ++seed) {
        const Observation o = env.Reset(seed);
        const float altitude = o[0] * 100.0f;
        EXPECT_GE(altitude, 15.0f - 1e-3f) << "seed " << seed;
        EXPECT_LE(altitude, 200.0f + 1e-3f) << "seed " << seed;
    }
}

// With the wind sensor off, the observation keeps its 18-value layout.
TEST(Env, DefaultObservationIsEighteenValues) {
    LandingEnv env;
    EXPECT_EQ(env.Reset(1).size(), 18u);
    EXPECT_EQ(env.ObservationSize(), 18u);
}

// With the wind sensor on, the observation grows to 21 values.
TEST(Env, WindSensorAddsThreeValues) {
    EnvConfig cfg;
    cfg.windSensor = true;
    LandingEnv env(cfg);
    EXPECT_EQ(env.Reset(1).size(), 21u);
}

// With no wind, the air-relative velocity sensor equals the body's velocity.
TEST(Env, WindSensorReadsBodyVelocityWhenCalm) {
    EnvConfig cfg;
    cfg.windSensor = true;
    cfg.atmosphere = true;
    LandingEnv env(cfg);

    env.ResetFromState({ 0.0f, 100.0f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(3.0f, -2.0f, 1.0f), glm::vec3(0.0f));
    const Observation o = env.GetObservation();

    EXPECT_NEAR(o[18], 3.0f / 50.0f, 1e-4f);
    EXPECT_NEAR(o[19], -2.0f / 50.0f, 1e-4f);
    EXPECT_NEAR(o[20], 1.0f / 50.0f, 1e-4f);
}

// With a steady 10 m/s base wind and a rocket at rest, the sensor reads the air
// moving past it: -10 m/s along X, shown as -10 / 50 in the observation.
// Steady 10 m/s wind along +X, rocket at rest: the air moves past the rocket at -10 m/s
// relative to it, shown as -10 / 50 in the observation.
TEST(Env, WindSensorReadsAirVelocityInWind) {
    EnvConfig cfg;
    cfg.windSensor = true;
    cfg.atmosphere = true;
    cfg.baseWind = { 10.0f, 0.0f, 0.0f };
    LandingEnv env(cfg);

    env.ResetFromState({ 0.0f, 100.0f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));
    const Observation o = env.GetObservation();

    EXPECT_NEAR(o[18], -10.0f / 50.0f, 1e-4f);
}

// Same seed with wind on gives the same wind history, so the same final state.
TEST(Env, WindEpisodeIsReproducible) {
    EnvConfig cfg;
    cfg.atmosphere = true;
    cfg.maxWindSpeed = 6.0f;
    cfg.gustInterval = 2.0f;

    auto finalObservation = [&](std::uint32_t seed) {
        LandingEnv env(cfg);
        env.Reset(seed);
        Observation last;
        for (int i = 0; i < 50; ++i) last = env.Step({ 0.0f, 0.0f, 0.0f }).observation;
        return last;
        };

    EXPECT_EQ(finalObservation(9), finalObservation(9));
}

// Runs the baseline until the episode ends, or the policy-step limit is reached.
namespace {

    StepResult RunBaseline(LandingEnv& env, const LandingBaseline& pilot, int maxPolicySteps) {
        StepResult r;
        for (int i = 0; i < maxPolicySteps; ++i) {
            r = env.Step(pilot.Compute(env.Rocket()));
            if (r.terminated || r.truncated) break;
        }
        return r;
    }

}  // namespace

// The sign check for the baseline: a 10 degree tilt must shrink within 1 s.
// The sign check for the baseline: a 10 degree tilt must shrink. The attitude loop has a
// time constant of about 2 s, so the check runs for 3 s, the same window as the attitude test.
TEST(Baseline, ReducesTiltFromTenDegrees) {
    LandingEnv env;
    env.ResetFromLowestPoint(100.0f, 0.0f, 0.0f,
        glm::angleAxis(glm::radians(10.0f), glm::vec3(1.0f, 0.0f, 0.0f)),
        glm::vec3(0.0f), glm::vec3(0.0f));
    LandingBaseline pilot;

    for (int i = 0; i < 90; ++i) env.Step(pilot.Compute(env.Rocket()));  // 90 x 4 = 360 steps = 3 s

    EXPECT_LT(env.Rocket().GetTiltDegrees(), 5.0f);
}

// An upright drop from 100 m must end as a soft landing.
TEST(Baseline, LandsUprightFromHeight) {
    LandingEnv env;
    env.ResetFromLowestPoint(100.0f, 0.0f, 0.0f, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));
    LandingBaseline pilot;

    const StepResult r = RunBaseline(env, pilot, 1800);  // 60 s cap

    EXPECT_EQ(r.status, LandingStatus::Landed);
    EXPECT_LT(env.Rocket().GetTouchdownSpeed(), kSoftLandingSpeed);
    EXPECT_LT(env.Rocket().GetTouchdownTiltDeg(), kSoftLandingTiltDeg);
}

// Success rate over 100 seeds, upright still starts at random heights. This is the bar
// the trained policy has to beat. The threshold is a goal, and the printed count tells you
// how far off you are if it fails.
TEST(Baseline, SuccessRateUprightStarts) {
    EnvConfig cfg;
    cfg.difficulty = 0.0f;
    LandingBaseline pilot;

    const int total = 100;
    int landed = 0;
    for (std::uint32_t seed = 1; seed <= static_cast<std::uint32_t>(total); ++seed) {
        LandingEnv env(cfg);
        env.Reset(seed);
        if (RunBaseline(env, pilot, 1800).status == LandingStatus::Landed) ++landed;
    }

    std::printf("Baseline upright success: %d / %d\n", landed, total);
    EXPECT_GE(landed, 90);
}

// Full random starts: report only, no pass or fail. This is the harder benchmark, and the
// policy's full-difficulty result will be compared against this number.
TEST(Baseline, SuccessRateFullRandomStarts) {
    EnvConfig cfg;
    cfg.difficulty = 1.0f;
    LandingBaseline pilot;

    const int total = 100;
    int landed = 0;
    for (std::uint32_t seed = 1; seed <= static_cast<std::uint32_t>(total); ++seed) {
        LandingEnv env(cfg);
        env.Reset(seed);
        if (RunBaseline(env, pilot, 1800).status == LandingStatus::Landed) ++landed;
    }

    std::printf("Baseline full-random success: %d / %d\n", landed, total);
}

// Diagnostic: bins the full-random results by starting tilt, to show where the baseline fails.
TEST(Baseline, DiagnoseFailuresByStartTilt) {
    EnvConfig cfg;
    cfg.difficulty = 1.0f;
    LandingBaseline pilot;

    int total[4] = { 0, 0, 0, 0 };
    int landed[4] = { 0, 0, 0, 0 };
    for (std::uint32_t seed = 1; seed <= 100; ++seed) {
        LandingEnv env(cfg);
        env.Reset(seed);
        const float tilt = env.Rocket().GetTiltDegrees();
        const int bin = std::min(3, static_cast<int>(tilt / 45.0f));
        const StepResult r = RunBaseline(env, pilot, 1800);
        ++total[bin];
        if (r.status == LandingStatus::Landed) ++landed[bin];
    }

    for (int b = 0; b < 4; ++b) {
        std::printf("start tilt %3d-%3d deg: %d / %d landed\n", b * 45, b * 45 + 45, landed[b], total[b]);
    }
}

// Hovering at 100 m with no feedback reaches the time limit, and the final step must carry
// the timeout penalty. The loop allows extra steps in case float rounding delays the cutoff.
TEST(Reward, TimeoutPenaltyApplied) {
    LandingEnv env;
    env.ResetFromState({ 0.0f, 104.0f, 0.0f }, glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
        glm::vec3(0.0f), glm::vec3(0.0f));

    const Action hover{ -0.2152f, 0.0f, 0.0f };
    StepResult r;
    for (int i = 0; i < 1900; ++i) {
        r = env.Step(hover);
        if (r.truncated) break;
    }

    EXPECT_TRUE(r.truncated);
    EXPECT_LT(r.reward, -40.0f);
}

// At the same altitude, a fast fall must earn less reward for the step than a slow one.
// The altitude shaping pays a little for the extra height lost, but the envelope penalty
// for 20 m/s at 30 m is far larger.
TEST(Reward, FastDescentEarnsLessThanSlowAtSameHeight) {
    LandingEnv slow, fast;
    const glm::quat upright(1.0f, 0.0f, 0.0f, 0.0f);
    slow.ResetFromState({ 0.0f, 30.0f, 0.0f }, upright, glm::vec3(0.0f), glm::vec3(0.0f));
    fast.ResetFromState({ 0.0f, 30.0f, 0.0f }, upright, { 0.0f, -20.0f, 0.0f }, glm::vec3(0.0f));

    const float rSlow = slow.Step(kCoast).reward;
    const float rFast = fast.Step(kCoast).reward;

    EXPECT_LT(rFast, rSlow);
}