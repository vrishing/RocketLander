#include "rocketphys/World.h"

#include <algorithm>
#include <cmath>

namespace rocketphys {

    namespace {

        // Below this upward speed after a bounce we just stop the body (prevents endless micro-bouncing).
        constexpr float kRestSpeed = 0.15f;

        // Applies an impulse acting at lever arm r from the center of mass.
        // Changes linear velocity by impulse / m, and angular velocity by I^-1 * (r x impulse).
        void ApplyImpulseAt(RigidBody& body, const glm::vec3& impulse, const glm::vec3& r,
            const glm::mat3& invInertiaWorld) {
            body.SetVelocity(body.GetVelocity() + impulse * body.GetInvMass());
            body.SetAngularVelocity(body.GetAngularVelocity() + invInertiaWorld * glm::cross(r, impulse));
        }

    }  // namespace

    std::size_t World::AddBody(const RigidBody& body) {
        bodies_.push_back(body);
        return bodies_.size() - 1;
    }

    void World::Step() {
        for (RigidBody& body : bodies_) {
            body.ApplyThrusters();                           // 1. thrust (throttle -> force)
            if (body.GetInvMass() > 0.0f) {
                body.ApplyForce(gravity_ * body.GetMass());  // 2. gravity, F = m * g
            }
            body.Integrate(kFixedDt);                        // 3. move and rotate
            ResolveGroundContact(body);                      // 4. ground
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

        const glm::vec3 n(0.0f, 1.0f, 0.0f);  // floor normal
        const glm::mat3 R = body.GetRotationMatrix();
        const glm::mat3 invInertiaWorld = R * body.GetInvInertiaBody() * glm::transpose(R);

        // 2. Velocity of the contact point = linear velocity + omega x r.
        glm::vec3 pointVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
        const float approachSpeed = -pointVel.y;  // positive when moving into the floor

        // 3. Record the first touchdown BEFORE the impulses below change the speed.
        body.RecordTouchdown(std::max(approachSpeed, 0.0f), body.GetTiltDegrees());

        if (approachSpeed <= 0.0f) return;  // already moving away: no contact force

        // 4. Normal impulse. Restitution: tiny impacts stop dead instead of micro-bouncing.
        const float e = (approachSpeed < kRestSpeed) ? 0.0f : restitution_;
        const glm::vec3 rxn = glm::cross(r, n);
        const float denomN = body.GetInvMass() + glm::dot(n, glm::cross(invInertiaWorld * rxn, r));
        if (denomN <= 0.0f) return;

        const float jn = (1.0f + e) * approachSpeed / denomN;  // normal impulse magnitude, N*s
        ApplyImpulseAt(body, jn * n, r, invInertiaWorld);

        // 5. Friction. After the normal impulse, the contact point's sideways (floor-plane)
        //    sliding speed is reduced toward zero, but the friction impulse is capped at
        //    mu * jn (Coulomb friction). Gripping works, and sliding cannot add energy.
        pointVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
        const glm::vec3 slide(pointVel.x, 0.0f, pointVel.z);
        const float slideSpeed = glm::length(slide);
        if (slideSpeed < 1e-6f || friction_ <= 0.0f) return;  // not sliding, or no friction

        const glm::vec3 T = slide / slideSpeed;  // unit direction of the slide
        const glm::vec3 rxT = glm::cross(r, T);
        const float denomT = body.GetInvMass() + glm::dot(T, glm::cross(invInertiaWorld * rxT, r));
        if (denomT <= 0.0f) return;

        // Impulse along T that would bring the sliding speed to zero. It is negative
        // because it opposes the slide, and it is capped by the friction limit.
        const float maxFriction = friction_ * jn;
        const float lambda = std::clamp(-slideSpeed / denomT, -maxFriction, maxFriction);
        ApplyImpulseAt(body, lambda * T, r, invInertiaWorld);
    }

}  // namespace rocketphys