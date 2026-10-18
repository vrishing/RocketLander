#include <gtest/gtest.h>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Hover test: a proportional-derivative height controller holds the rocket
// at a target altitude. Controller lives in the test, not in the library.
//
//   throttle = hover + kp * (target - y) - kd * vy
//
// hover = weight / max thrust = 9810 / 25000 = 0.3924
// kp = 0.002 -> stiffness 50 N/m
// kd = 0.02  -> damping 500 N*s/m (overdamped, so it settles without oscillating)
// ---------------------------------------------------------------------------
TEST(Hover, MainEngineHoldsTargetAltitude) {
    const float target = 1000.0f;
    const float hover = 9810.0f / 25000.0f;
    const float kp = 0.002f;
    const float kd = 0.02f;

    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, target + 2.0f, 0.0f });  // start 2 m above target
    const std::size_t id = world.AddBody(rocket);

    // 40 s of simulated time: 40 / (1/120) = 4800 steps.
    for (int step = 0; step < 4800; ++step) {
        RigidBody& body = world.GetBody(id);
        const float error = target - body.GetPosition().y;
        const float vy = body.GetVelocity().y;
        body.SetThrottle(kMainEngine, hover + kp * error - kd * vy);
        world.Step();
    }

    const RigidBody& body = world.GetBody(id);
    EXPECT_NEAR(body.GetPosition().y, target, 0.05f);     // within 5 cm
    EXPECT_NEAR(body.GetVelocity().y, 0.0f, 0.01f);       // essentially at rest
    EXPECT_NEAR(body.GetPosition().x, 0.0f, 1e-3f);       // no sideways drift
    EXPECT_NEAR(body.GetPosition().z, 0.0f, 1e-3f);
    EXPECT_NEAR(glm::length(body.GetAngularVelocity()), 0.0f, 1e-3f);  // no spin
}

// ---------------------------------------------------------------------------
// Attitude: a tilt is a disturbance, not an error the rocket corrects by itself.
// The main engine sits on the body's axis, so it produces no torque. With only
// hover throttle, a rocket tilted 10 degrees keeps exactly that tilt.
// ---------------------------------------------------------------------------
TEST(Attitude, TiltedRocketDoesNotSelfCorrect) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 1000.0f, 0.0f });
    rocket.SetOrientation(glm::angleAxis(glm::radians(10.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    const std::size_t id = world.AddBody(rocket);
    world.GetBody(id).SetThrottle(kMainEngine, 9810.0f / 25000.0f);

    RunSteps(world, 120);  // 1 s

    // Angle between the body's up axis and world up. For a unit vector, angle = acos(y).
    const glm::vec3 up = world.GetBody(id).GetOrientation() * glm::vec3(0.0f, 1.0f, 0.0f);
    const float tiltDeg = glm::degrees(std::acos(glm::clamp(up.y, -1.0f, 1.0f)));
    EXPECT_NEAR(tiltDeg, 10.0f, 0.01f);
}


// ---------------------------------------------------------------------------
// Attitude controller tests.
// ---------------------------------------------------------------------------
namespace {

    // Angle between the rocket's up axis and world up, in degrees.
    float UpTiltDegrees(const RigidBody& rocket) {
        const glm::vec3 up = rocket.GetRotationMatrix() * glm::vec3(0.0f, 1.0f, 0.0f);
        return glm::degrees(std::acos(glm::clamp(up.y, -1.0f, 1.0f)));
    }

    // Runs hover throttle plus the attitude controller for `steps` physics steps.
    void RunHoverWithAttitude(World& world, std::size_t id, int steps) {
        const float hover = 9810.0f / 25000.0f;
        for (int i = 0; i < steps; ++i) {
            RigidBody& body = world.GetBody(id);
            body.SetThrottle(kMainEngine, hover);
            UpdateRocketAttitude(body);
            world.Step();
        }
    }

}  // namespace

// Sign test: a 10 degree tilt must get smaller, not bigger. If this fails, the
// torque sign is flipped. Fix that before tuning anything.
TEST(Attitude, ControllerReducesTilt) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 1000.0f, 0.0f });
    rocket.SetOrientation(glm::angleAxis(glm::radians(10.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    const std::size_t id = world.AddBody(rocket);

    RunHoverWithAttitude(world, id, 360);  // 3 s

    // Expected: the tilt is about 2 to 3 degrees, so 5 degrees is a safe upper bound.
    EXPECT_LT(UpTiltDegrees(world.GetBody(id)), 5.0f);
}

// Settling test: after 10 s the tilt should be almost gone.
TEST(Attitude, ControllerSettlesTiltNearUpright) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 1000.0f, 0.0f });
    rocket.SetOrientation(glm::angleAxis(glm::radians(10.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    const std::size_t id = world.AddBody(rocket);

    RunHoverWithAttitude(world, id, 1200);  // 10 s

    EXPECT_LT(UpTiltDegrees(world.GetBody(id)), 0.5f);
}

// ---------------------------------------------------------------------------
// Contact at the lowest point.
// ---------------------------------------------------------------------------

// A tilted rocket must never sink below the floor, and must end up tipped over.
TEST(Contact, TiltedRocketTipsOverOnFloor) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 5.0f, 0.0f });
    rocket.SetOrientation(glm::angleAxis(glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    const std::size_t id = world.AddBody(rocket);

    for (int i = 0; i < 1200; ++i) {  // 10 s
        world.Step();
        EXPECT_GE(world.GetBody(id).GetLowestPoint().y, -1e-3f);
    }

    const RigidBody& body = world.GetBody(id);
    EXPECT_NEAR(body.GetLowestPoint().y, 0.0f, 0.05f);  // resting on the floor
    EXPECT_GT(UpTiltDegrees(body), 30.0f);              // it tipped over
}

// An upright rocket must land and stay upright. This checks the contact code
// doesn't invent torque when the rocket is symmetric.
TEST(Contact, UprightRocketLandsAndStaysUpright) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 5.0f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    for (int i = 0; i < 1200; ++i) world.Step();  // 10 s

    const RigidBody& body = world.GetBody(id);
    EXPECT_NEAR(body.GetLowestPoint().y, 0.0f, 0.01f);
    EXPECT_LT(UpTiltDegrees(body), 0.1f);
}

// ---------------------------------------------------------------------------
// Landing detection.
// ---------------------------------------------------------------------------
namespace {

    // Steps the world until the rocket leaves Flying. Returns false if it never lands.
    bool RunUntilTouchdown(World& world, std::size_t id, int maxSteps) {
        for (int i = 0; i < maxSteps; ++i) {
            world.Step();
            if (world.GetBody(id).GetLandingStatus() != LandingStatus::Flying) return true;
        }
        return false;
    }

}  // namespace

// The classification rule on its own, at the limits.
TEST(Landing, ClassifyRulesMatchLimits) {
    EXPECT_EQ(ClassifyTouchdown(1.0f, 5.0f), LandingStatus::Landed);
    EXPECT_EQ(ClassifyTouchdown(3.0f, 5.0f), LandingStatus::Crashed);   // too fast
    EXPECT_EQ(ClassifyTouchdown(1.0f, 15.0f), LandingStatus::Crashed);  // too tilted
}

// Lowest point 5 cm above the floor, so impact speed is sqrt(2 * 9.81 * 0.05) = 0.99 m/s.
TEST(Landing, GentleUprightLandingIsLanded) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 4.05f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    ASSERT_TRUE(RunUntilTouchdown(world, id, 2400));  // 20 s cap

    const RigidBody& body = world.GetBody(id);
    EXPECT_EQ(body.GetLandingStatus(), LandingStatus::Landed);
    EXPECT_NEAR(body.GetTouchdownSpeed(), 0.99f, 0.1f);
    EXPECT_LT(body.GetTouchdownTiltDeg(), 0.1f);
}

// Dropped from 100 m: impact speed is about 43 m/s, far over the limit.
TEST(Landing, FastUprightDropIsCrashed) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 100.0f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    ASSERT_TRUE(RunUntilTouchdown(world, id, 4800));  // 40 s cap

    const RigidBody& body = world.GetBody(id);
    EXPECT_EQ(body.GetLandingStatus(), LandingStatus::Crashed);
    EXPECT_GT(body.GetTouchdownSpeed(), 40.0f);
}

// Tilted 30 degrees and dropped from 5 m. Tilt is set before any torque acts,
// so the recorded touchdown tilt should be the starting 30 degrees.
TEST(Landing, TiltedDropIsCrashed) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 5.0f, 0.0f });
    rocket.SetOrientation(glm::angleAxis(glm::radians(30.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
    const std::size_t id = world.AddBody(rocket);

    ASSERT_TRUE(RunUntilTouchdown(world, id, 2400));

    const RigidBody& body = world.GetBody(id);
    EXPECT_EQ(body.GetLandingStatus(), LandingStatus::Crashed);
    EXPECT_NEAR(body.GetTouchdownTiltDeg(), 30.0f, 0.5f);
}

// Status is recorded once. A second bounce must not overwrite the first touchdown.
TEST(Landing, StatusIsRecordedOnlyOnce) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 4.05f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    ASSERT_TRUE(RunUntilTouchdown(world, id, 2400));
    const float firstSpeed = world.GetBody(id).GetTouchdownSpeed();

    for (int i = 0; i < 1200; ++i) world.Step();  // 10 more seconds, including any bounces

    EXPECT_EQ(world.GetBody(id).GetTouchdownSpeed(), firstSpeed);
}
// ---------------------------------------------------------------------------
// Friction. The rocket lands with a 5 m/s sideways velocity, with no thrust.
// We measure the sliding speed of the FOOT (the contact point), not the center.
// The base sits 4 m below the center, so friction there tips the rocket over,
// and the center's speed changes for reasons other than sliding.
// ---------------------------------------------------------------------------
namespace {

    // Floor-plane speed of the lowest point: its velocity is v + omega x r.
    float FootSlideSpeed(const RigidBody& body) {
        const glm::vec3 r = body.GetLowestPoint() - body.GetPosition();
        const glm::vec3 footVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
        return std::sqrt(footVel.x * footVel.x + footVel.z * footVel.z);
    }

    // Drops the rocket with its foot 5 cm above the floor and a 5 m/s sideways velocity,
    // steps for `seconds`, and returns the foot's sliding speed.
    float FootSlideAfter(float mu, float seconds) {
        World world;
        world.SetGravity({ 0.0f, -9.81f, 0.0f });
        world.SetFriction(mu);

        RigidBody rocket = MakeRocketBody();
        rocket.SetPosition({ 0.0f, 4.05f, 0.0f });  // lowest point 5 cm above the floor
        rocket.SetVelocity({ 5.0f, 0.0f, 0.0f });
        const std::size_t id = world.AddBody(rocket);

        const int steps = static_cast<int>(seconds / World::kFixedDt + 0.5f);
        for (int i = 0; i < steps; ++i) world.Step();

        return FootSlideSpeed(world.GetBody(id));
    }

}  // namespace

// With no friction the foot keeps its sideways speed. This also checks that the
// normal impulse is purely vertical and adds no sideways effect.
TEST(Friction, FrictionlessFootKeepsSliding) {
    EXPECT_NEAR(FootSlideAfter(0.0f, 1.0f), 5.0f, 0.01f);
}

// With friction 0.6 the foot must stop slipping within 1 s.
// Friction can remove at most mu * g per second of sliding (about 5.9 m/s^2),
// so 5 m/s takes about 0.85 s to clear.
TEST(Friction, FrictionStopsFootSlide) {
    EXPECT_LT(FootSlideAfter(0.6f, 1.0f), 1.0f);
}

// A stronger coefficient must leave the foot sliding less than a weaker one.
TEST(Friction, StrongerFrictionSlowsFootMore) {
    const float strong = FootSlideAfter(0.6f, 0.3f);
    const float weak = FootSlideAfter(0.1f, 0.3f);
    EXPECT_LT(strong, weak);
}

// ---------------------------------------------------------------------------
// Atmosphere and air drag.
// ---------------------------------------------------------------------------
namespace {

    // Terminal speed for the rocket falling along its axis, from the balance
    // m * g = 1/2 * rho * Cd * (pi r^2) * v^2.
    float TerminalSpeedAt(const World& world, float height, float mass, float radius) {
        const float rho = world.AirDensityAt(height);
        const float area = 3.14159265f * radius * radius;
        return std::sqrt(2.0f * mass * 9.81f / (rho * world.GetDragCoefficient() * area));
    }

}  // namespace

// Vacuum is the default. A 1 s fall must match v = g * t exactly, as before.
TEST(Atmosphere, OffByDefaultFallsInVacuum) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });
    EXPECT_FALSE(world.HasAtmosphere());

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 1000.0f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    RunSteps(world, 120);  // 1 s

    EXPECT_NEAR(world.GetBody(id).GetVelocity().y, -9.81f, 0.01f);
}

// Density at sea level is 1.225 kg/m^3, and falls to 1/e of that one scale height up.
TEST(Atmosphere, DensityFallsWithAltitude) {
    World world;
    world.SetAtmosphere(true);
    EXPECT_NEAR(world.AirDensityAt(0.0f), 1.225f, 1e-4f);
    EXPECT_NEAR(world.AirDensityAt(8500.0f), 1.225f / std::exp(1.0f), 1e-3f);
}

// Sideways motion is slowed by drag. Expected: 0.5 * 1.089 * 0.8 * 8 m^2 * v^2 / 1000 kg
// at 1000 m, which is about 0.087 m/s^2 at 5 m/s, so the speed drops to about 4.91 m/s in 1 s.
TEST(Atmosphere, DragSlowsSidewaysMotion) {
    World world;
    world.SetGravity({ 0.0f, 0.0f, 0.0f });
    world.SetAtmosphere(true);

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, 1000.0f, 0.0f });
    rocket.SetVelocity({ 5.0f, 0.0f, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    RunSteps(world, 120);  // 1 s

    const glm::vec3 v = world.GetBody(id).GetVelocity();
    EXPECT_GT(v.x, 0.0f);                      // still moving the same way
    EXPECT_NEAR(v.x, 4.91f, 0.05f);            // slowed by drag
}

// Terminal speed is an equilibrium: a rocket already falling at it must not
// accelerate. The tolerance is 2% because the air thins by about 2% over the
// 170 m the rocket falls in that second.
TEST(Atmosphere, TerminalSpeedIsEquilibrium) {
    World world;
    world.SetGravity({ 0.0f, -9.81f, 0.0f });
    world.SetAtmosphere(true);

    const float height = 1000.0f;
    const float vt = TerminalSpeedAt(world, height, 1000.0f, 0.5f);  // about 169 m/s

    RigidBody rocket = MakeRocketBody();
    rocket.SetPosition({ 0.0f, height, 0.0f });
    rocket.SetVelocity({ 0.0f, -vt, 0.0f });
    const std::size_t id = world.AddBody(rocket);

    RunSteps(world, 120);  // 1 s

    EXPECT_NEAR(world.GetBody(id).GetVelocity().y, -vt, 0.02f * vt);
}