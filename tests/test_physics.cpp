#include <gtest/gtest.h>

#include <glm/glm.hpp>

#include "rocketphys/RigidBody.h"
#include "rocketphys/World.h"

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
