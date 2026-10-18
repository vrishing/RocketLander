#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "rocketphys/LandingEnv.h"

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