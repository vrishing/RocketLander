#include <gtest/gtest.h>

#include <glm/glm.hpp>

#include "rocketphys/RigidBody.h"
#include "rocketphys/World.h"
#include "rocketphys/Rocket.h"

using namespace rocketphys;

namespace {

// Helper: a 1 kg unit box with no damping, high in the air so the ground never interferes.
RigidBody MakeFreeBody(const glm::vec3& pos, float mass = 1.0f) {
    RigidBody b;
    b.SetMass(mass);
    b.SetBoxInertia(1.0f, 1.0f, 1.0f);
    b.SetDamping(1.0f, 1.0f);
    b.SetPosition(pos);
    return b;
}

}  // namespace

// 1) Free fall for 1 s should match 0.5 * g * t^2 (semi-implicit Euler is slightly off, hence 0.1).
TEST(Physics, FreeFallMatchesFormula) {
    World world;
    world.SetGravity({0.0f, -9.81f, 0.0f});
    const std::size_t id = world.AddBody(MakeFreeBody({0.0f, 1000.0f, 0.0f}));

    for (int i = 0; i < 120; ++i) world.Step();  // 120 steps * (1/120 s) = 1 s

    const float expectedY = 1000.0f - 0.5f * 9.81f * 1.0f * 1.0f;
    EXPECT_NEAR(world.GetBody(id).GetPosition().y, expectedY, 0.1f);
    EXPECT_NEAR(world.GetBody(id).GetVelocity().y, -9.81f, 0.01f);
}

// 2) With no gravity and no forces, velocity must not change (Newton's first law).
TEST(Physics, ZeroGravityKeepsConstantVelocity) {
    World world;
    world.SetGravity({0.0f, 0.0f, 0.0f});
    RigidBody b = MakeFreeBody({0.0f, 100.0f, 0.0f});
    b.SetVelocity({1.0f, 2.0f, -3.0f});
    const std::size_t id = world.AddBody(b);

    for (int i = 0; i < 240; ++i) world.Step();  // 2 s

    const glm::vec3 p = world.GetBody(id).GetPosition();
    EXPECT_NEAR(p.x, 2.0f, 1e-3f);
    EXPECT_NEAR(p.y, 100.0f + 4.0f, 1e-3f);
    EXPECT_NEAR(p.z, -6.0f, 1e-3f);
}

// 3) The orientation quaternion must stay unit length even after thousands of rotation steps.
TEST(Physics, QuaternionStaysNormalized) {
    World world;
    world.SetGravity({0.0f, 0.0f, 0.0f});
    RigidBody b = MakeFreeBody({0.0f, 100.0f, 0.0f});
    b.SetAngularVelocity({1.0f, 2.0f, 3.0f});
    const std::size_t id = world.AddBody(b);

    for (int i = 0; i < 10000; ++i) world.Step();

    EXPECT_NEAR(glm::length(world.GetBody(id).GetOrientation()), 1.0f, 1e-4f);
}

// 4) Same inputs => bit-identical outputs. RL training and debugging depend on this.
TEST(Physics, SimulationIsDeterministic) {
    auto run = []() {
        World world;
        const std::size_t id = world.AddBody(MakeFreeBody({0.0f, 50.0f, 0.0f}));
        for (int i = 0; i < 2000; ++i) {
            RigidBody& body = world.GetBody(id);
            body.ApplyForceAtPoint({0.0f, 12.0f, 0.0f}, body.GetPosition() + glm::vec3(0.3f, 0.0f, 0.0f));
            world.Step();
        }
        return world.GetBody(id);
    };

    const RigidBody a = run();
    const RigidBody b = run();
    EXPECT_EQ(a.GetPosition(), b.GetPosition());
    EXPECT_EQ(a.GetOrientation(), b.GetOrientation());
}

// 5) A constant torque on a box gives omega = torque / I * t (checks your inertia formula).
TEST(Physics, TorqueUsesInertiaTensor) {
    World world;
    world.SetGravity({0.0f, 0.0f, 0.0f});
    RigidBody b;
    b.SetMass(10.0f);
    b.SetBoxInertia(1.0f, 2.0f, 3.0f);  // width, height, depth
    b.SetDamping(1.0f, 1.0f);
    b.SetPosition({0.0f, 100.0f, 0.0f});
    const std::size_t id = world.AddBody(b);

    const float iyy = 10.0f / 12.0f * (1.0f * 1.0f + 3.0f * 3.0f);  // m/12 * (w^2 + d^2)
    for (int i = 0; i < 120; ++i) {                                  // 1 s
        world.GetBody(id).ApplyTorque({0.0f, 5.0f, 0.0f});
        world.Step();
    }

    const glm::vec3 w = world.GetBody(id).GetAngularVelocity();
    EXPECT_NEAR(w.y, 5.0f / iyy * 1.0f, 1e-3f);
    EXPECT_NEAR(w.x, 0.0f, 1e-4f);
    EXPECT_NEAR(w.z, 0.0f, 1e-4f);
}

// 6) A dropped body must never end up under the floor and should come to rest on it.
TEST(Physics, GroundStopsFallingBody) {
    World world;
    world.SetRestitution(0.5f);
    RigidBody b = MakeFreeBody({0.0f, 2.0f, 0.0f});
    b.SetHalfHeight(0.5f);
    const std::size_t id = world.AddBody(b);

    for (int i = 0; i < 1200; ++i) {  // 10 s
        world.Step();
        EXPECT_GE(world.GetBody(id).GetPosition().y, 0.5f - 1e-4f);
    }
    EXPECT_NEAR(world.GetBody(id).GetPosition().y, 0.5f, 1e-3f);
    EXPECT_NEAR(world.GetBody(id).GetVelocity().y, 0.0f, 0.2f);
}


// ---------------------------------------------------------------------------
// Thruster tests. Rocket: 1000 kg, r = 0.5 m, h = 8 m. Main engine 25 kN, RCS 1 kN.
// ---------------------------------------------------------------------------
namespace {

    // Pitch/roll inertia of the cylinder: m * (3r^2 + h^2) / 12  (about 5395.83 kg*m^2)
    constexpr float kInertiaSideways = 1000.0f * (3.0f * 0.25f + 64.0f) / 12.0f;

    std::size_t AddRocket(World& world, const glm::vec3& position) {
        RigidBody rocket = MakeRocketBody();
        rocket.SetPosition(position);
        return world.AddBody(rocket);
    }

    void RunSteps(World& world, int steps) {
        for (int i = 0; i < steps; ++i) world.Step();
    }

}  // namespace

// Main engine at full throttle: a = (25000 - 1000 * 9.81) / 1000 = 15.19 m/s^2.
// The thrust passes through the center of mass, so there must be no spin.
TEST(Thrusters, MainEngineAcceleratesUpwardWithoutSpin) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });
    const std::size_t id = AddRocket(world, { 0.0f, 1000.0f, 0.0f });
    world.GetBody(id).SetThrottle(kMainEngine, 1.0f);

    RunSteps(world, 120);  // 1 s

    const RigidBody& b = world.GetBody(id);
    EXPECT_NEAR(b.GetVelocity().y, 15.19f, 0.01f);
    EXPECT_NEAR(b.GetVelocity().x, 0.0f, 1e-3f);
    EXPECT_NEAR(b.GetVelocity().z, 0.0f, 1e-3f);
    EXPECT_NEAR(glm::length(b.GetAngularVelocity()), 0.0f, 1e-4f);
}

// Hover throttle = weight / max thrust = 9810 / 25000 = 0.3924.
TEST(Thrusters, HoverThrottleCancelsGravity) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });
    const std::size_t id = AddRocket(world, { 0.0f, 1000.0f, 0.0f });
    world.GetBody(id).SetThrottle(kMainEngine, 9810.0f / 25000.0f);

    RunSteps(world, 120);

    EXPECT_NEAR(world.GetBody(id).GetVelocity().y, 0.0f, 0.01f);
}

// Throttle must be clamped: 5.0 behaves like 1.0, and -3.0 behaves like 0.
TEST(Thrusters, ThrottleIsClamped) {
    World world;
    world.SetGravity({ 0.0f, 0.0f, 0.0f });
    const std::size_t over = AddRocket(world, { 0.0f, 1000.0f, 0.0f });
    const std::size_t under = AddRocket(world, { 50.0f, 1000.0f, 0.0f });
    world.GetBody(over).SetThrottle(kMainEngine, 5.0f);
    world.GetBody(under).SetThrottle(kMainEngine, -3.0f);

    RunSteps(world, 120);

    EXPECT_NEAR(world.GetBody(over).GetVelocity().y, 25.0f, 0.01f);  // 25000 N / 1000 kg * 1 s
    EXPECT_NEAR(world.GetBody(under).GetVelocity().y, 0.0f, 1e-4f);
}

// One RCS thruster alone pushes AND spins.
// Pitch A top: F = 1000 N along -Z at (0, 3, 0.5)  ->  torque (-3000, 0, 0).
// After 0.1 s: vz = -0.1 m/s, wx = -3000 / I * 0.1.
TEST(Thrusters, SingleRcsThrusterPushesAndSpins) {
    World world;
    world.SetGravity({ 0.0f, 0.0f, 0.0f });
    const std::size_t id = AddRocket(world, { 0.0f, 1000.0f, 0.0f });
    world.GetBody(id).SetThrottle(kPitchATop, 1.0f);

    RunSteps(world, 12);  // 0.1 s (short, so the body barely tilts)

    const RigidBody& b = world.GetBody(id);
    const float expectedWx = -3000.0f / kInertiaSideways * 0.1f;  // about -0.0556 rad/s
    EXPECT_NEAR(b.GetVelocity().z, -0.1f, 1e-3f);
    EXPECT_NEAR(b.GetVelocity().y, 0.0f, 1e-3f);
    EXPECT_NEAR(b.GetAngularVelocity().x, expectedWx, std::abs(expectedWx) * 0.01f);
    EXPECT_NEAR(b.GetAngularVelocity().y, 0.0f, 1e-4f);
    EXPECT_NEAR(b.GetAngularVelocity().z, 0.0f, 1e-4f);
}

// Each diagonal pair (one thruster on the top ring + one on the bottom ring) must
// give a pure torque: zero net force, and 2 * 3000 N*m about the expected axis.
TEST(Thrusters, DiagonalPairsGivePureTorque) {
    struct PairCase {
        std::size_t first, second;
        glm::vec3 axis;  // expected direction of angular velocity
    };
    const PairCase cases[] = {
        { kPitchATop, kPitchABottom,{ -1.0f, 0.0f, 0.0f } },
        { kPitchBTop, kPitchBBottom,{ +1.0f, 0.0f, 0.0f } },
        { kRollATop, kRollABottom,{ 0.0f, 0.0f, +1.0f } },
        { kRollBTop, kRollBBottom,{ 0.0f, 0.0f, -1.0f } },
    };

    for (std::size_t i = 0; i < 4; ++i) {
        SCOPED_TRACE(testing::Message() << "pair case " << i);
        World world;
        world.SetGravity({ 0.0f, 0.0f, 0.0f });
        const std::size_t id = AddRocket(world, { 0.0f, 1000.0f, 0.0f });
        world.GetBody(id).SetThrottle(cases[i].first, 1.0f);
        world.GetBody(id).SetThrottle(cases[i].second, 1.0f);

        RunSteps(world, 12);  // 0.1 s

        const RigidBody& b = world.GetBody(id);
        const glm::vec3 expectedW = cases[i].axis * (6000.0f / kInertiaSideways * 0.1f);  // about 0.111 rad/s
        EXPECT_NEAR(glm::length(b.GetVelocity()), 0.0f, 1e-3f);  // no net force
        EXPECT_NEAR(b.GetAngularVelocity().x, expectedW.x, 1e-3f);
        EXPECT_NEAR(b.GetAngularVelocity().y, expectedW.y, 1e-3f);
        EXPECT_NEAR(b.GetAngularVelocity().z, expectedW.z, 1e-3f);
    }
}