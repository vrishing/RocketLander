#include "rocketphys/World.h"

#include <algorithm>
#include <cmath>

namespace rocketphys {

    namespace {
        // Below this upward speed after a bounce we just stop the body (prevents endless micro-bouncing).
        constexpr float kRestSpeed = 0.15f;
    }  // namespace

    std::size_t World::AddBody(const RigidBody& body) {
        bodies_.push_back(body);
        return bodies_.size() - 1;
    }

    void World::Step() {
        for (RigidBody& body : bodies_) {
            body.ApplyThrusters();                          // 1. thrust (throttle -> force)
            if (body.GetInvMass() > 0.0f) {
                body.ApplyForce(gravity_ * body.GetMass()); // 2. gravity, F = m * g
            }
            body.Integrate(kFixedDt);                       // 3. move and rotate
            ResolveGroundContact(body);                     // 4. ground
        }
        ++stepCount_;
    }

    void World::ResolveGroundContact(RigidBody& body) const {
        if (body.GetInvMass() == 0.0f) return;  // static bodies don't respond

        const glm::vec3 center = body.GetPosition();
        const glm::vec3 contact = body.GetLowestPoint();
        if (contact.y >= 0.0f) return;          // lowest point is above the floor

        // Lever arm from the center of mass to the contact point.
        const glm::vec3 r = contact - center;

        // 1. Lift the body so the contact point sits exactly on the floor.
        body.SetPosition(center + glm::vec3(0.0f, -contact.y, 0.0f));

        // 2. Velocity of the contact point = linear velocity + omega x r.
        const glm::vec3 n(0.0f, 1.0f, 0.0f);  // floor normal
        const glm::vec3 pointVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
        const float approachSpeed = -pointVel.y;  // positive when moving into the floor

        // 3. Record the first touchdown BEFORE the impulse below changes the speed.
        body.RecordTouchdown(std::max(approachSpeed, 0.0f), body.GetTiltDegrees());

        if (approachSpeed <= 0.0f) return;  // already moving away

        // 4. Restitution. Tiny impacts stop dead instead of micro-bouncing.
        const float e = (approachSpeed < kRestSpeed) ? 0.0f : restitution_;

        // 5. Effective inverse mass for an impulse at r along n. The rotational term
        //    means an off-center hit spins the body as well as slowing it.
        const glm::mat3 R = body.GetRotationMatrix();
        const glm::mat3 invInertiaWorld = R * body.GetInvInertiaBody() * glm::transpose(R);
        const glm::vec3 rxn = glm::cross(r, n);
        const float denom = body.GetInvMass() + glm::dot(n, glm::cross(invInertiaWorld * rxn, r));
        if (denom <= 0.0f) return;

        // 6. Impulse magnitude, applied to linear and angular velocity.
        const float j = (1.0f + e) * approachSpeed / denom;
        const glm::vec3 impulse = j * n;
        body.SetVelocity(body.GetVelocity() + impulse * body.GetInvMass());
        body.SetAngularVelocity(body.GetAngularVelocity() + invInertiaWorld * glm::cross(r, impulse));
    }

}  // namespace rocketphys