#include "rocketphys/World.h"
#include "rocketphys/RigidBody.h"
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
        body.ApplyThrusters();
        if (body.GetInvMass() > 0.0f) {
            body.ApplyForce(gravity_ * body.GetMass());  // F = m * g
        }
        body.Integrate(kFixedDt);
        ResolveGroundContact(body);
    }
    ++stepCount_;
}

void World::ResolveGroundContact(RigidBody& body) const {
    // Ground plane is y = 0. The body's lowest point is position.y - halfHeight.
    glm::vec3 pos = body.GetPosition();
    if (pos.y - body.GetHalfHeight() > 0.0f) return;

    pos.y = body.GetHalfHeight();  // snap back on top of the floor
    body.SetPosition(pos);

    glm::vec3 vel = body.GetVelocity();
    if (vel.y < 0.0f) {
        vel.y = std::fabs(vel.y) * restitution_;  // bounce
        if (vel.y < kRestSpeed) vel.y = 0.0f;
        body.SetVelocity(vel);
    }
    // TODO (hour 1.5 -> landing): report Landed/Crashed using touchdown speed and tilt.
}

}  // namespace rocketphys
