#include "rocketphys/World.h"

#include <algorithm>
#include <cmath>

namespace rocketphys {

    namespace {

        // Below this upward speed after a bounce we just stop the body (prevents endless micro-bouncing).
        constexpr float kRestSpeed = 0.15f;

        // Scale height of the exponential atmosphere, m.
        constexpr float kScaleHeight = 8500.0f;

        // Contact passes per step when legs are out. Feet that touch together need a few passes to settle.
        constexpr int kLegContactPasses = 3;

        // Gusts ease toward their target with this time constant, s. Keeps wind changes smooth.
        constexpr float kGustRampTime = 1.0f;

        constexpr float kTwoPi = 6.28318531f;

        // Applies an impulse acting at lever arm r from the center of mass.
        void ApplyImpulseAt(RigidBody& body, const glm::vec3& impulse, const glm::vec3& r,
            const glm::mat3& invInertiaWorld) {
            body.SetVelocity(body.GetVelocity() + impulse * body.GetInvMass());
            body.SetAngularVelocity(body.GetAngularVelocity() + invInertiaWorld * glm::cross(r, impulse));
        }

        // Resolves one contact point against the floor (y = 0).
        // legIndex < 0 means the base rim; otherwise it is that leg's foot.
        void ResolveContactPoint(RigidBody& body, int legIndex, float restitution, float friction) {
            const glm::vec3 contact = (legIndex < 0)
                ? body.GetLowestPoint()
                : body.GetLegFootWorld(static_cast<std::size_t>(legIndex));
            if (contact.y >= 0.0f) return;  // this point is above the floor

            const glm::vec3 center = body.GetPosition();
            const glm::vec3 r = contact - center;

            // 1. Lift the body so the contact point sits exactly on the floor.
            body.SetPosition(center + glm::vec3(0.0f, -contact.y, 0.0f));

            const glm::vec3 n(0.0f, 1.0f, 0.0f);  // floor normal
            const glm::mat3 R = body.GetRotationMatrix();
            const glm::mat3 invInertiaWorld = R * body.GetInvInertiaBody() * glm::transpose(R);

            // 2. Velocity of the contact point = linear velocity + omega x r.
            glm::vec3 pointVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
            const float approachSpeed = -pointVel.y;

            // 3. Record the first touchdown BEFORE the impulses below change the speed.
            body.RecordTouchdown(std::max(approachSpeed, 0.0f), body.GetTiltDegrees());

            if (approachSpeed <= 0.0f) return;  // already moving away

            // 4. Normal impulse. Restitution: tiny impacts stop dead instead of micro-bouncing.
            const float e = (approachSpeed < kRestSpeed) ? 0.0f : restitution;
            const glm::vec3 rxn = glm::cross(r, n);
            const float denomN = body.GetInvMass() + glm::dot(n, glm::cross(invInertiaWorld * rxn, r));
            if (denomN <= 0.0f) return;

            const float jn = (1.0f + e) * approachSpeed / denomN;
            ApplyImpulseAt(body, jn * n, r, invInertiaWorld);

            // 5. Friction, capped at mu * jn (Coulomb friction).
            pointVel = body.GetVelocity() + glm::cross(body.GetAngularVelocity(), r);
            const glm::vec3 slide(pointVel.x, 0.0f, pointVel.z);
            const float slideSpeed = glm::length(slide);
            if (slideSpeed < 1e-6f || friction <= 0.0f) return;

            const glm::vec3 T = slide / slideSpeed;
            const glm::vec3 rxT = glm::cross(r, T);
            const float denomT = body.GetInvMass() + glm::dot(T, glm::cross(invInertiaWorld * rxT, r));
            if (denomT <= 0.0f) return;

            const float maxFriction = friction * jn;
            const float lambda = std::clamp(-slideSpeed / denomT, -maxFriction, maxFriction);
            ApplyImpulseAt(body, lambda * T, r, invInertiaWorld);
        }

    }  // namespace

    std::size_t World::AddBody(const RigidBody& body) {
        bodies_.push_back(body);
        return bodies_.size() - 1;
    }

    void World::SetAtmosphere(bool enabled, float seaLevelDensity) {
        atmosphere_ = enabled;
        seaLevelDensity_ = (seaLevelDensity < 0.0f) ? 0.0f : seaLevelDensity;
    }

    float World::AirDensityAt(float height) const {
        const float h = (height < 0.0f) ? 0.0f : height;
        return seaLevelDensity_ * std::exp(-h / kScaleHeight);
    }

    void World::SetGusts(bool enabled, float maxGustSpeed, float gustInterval, std::uint32_t seed) {
        gustsEnabled_ = enabled;
        maxGustSpeed_ = std::max(0.0f, maxGustSpeed);
        gustInterval_ = std::max(0.1f, gustInterval);  // a zero interval would never advance
        rng_.seed(seed);
        gustTarget_ = glm::vec3(0.0f);
        gustNow_ = glm::vec3(0.0f);
        nextGustTime_ = 0.0;  // the first gust is chosen on the next step
    }

    void World::UpdateGusts() {
        if (!gustsEnabled_) {
            gustNow_ = glm::vec3(0.0f);
            return;
        }

        // Pick a new target when the interval has elapsed.
        if (GetTime() >= nextGustTime_) {
            std::uniform_real_distribution<float> angle(0.0f, kTwoPi);
            std::uniform_real_distribution<float> speed(0.0f, maxGustSpeed_);
            const float a = angle(rng_);
            const float s = speed(rng_);
            gustTarget_ = glm::vec3(s * std::cos(a), 0.0f, s * std::sin(a));
            nextGustTime_ += gustInterval_;
        }

        // Ease toward the target. The step is a fraction of the gap, so the wind changes smoothly.
        const float blend = 1.0f - std::exp(-kFixedDt / kGustRampTime);
        gustNow_ += (gustTarget_ - gustNow_) * blend;
    }

    void World::Step() {
        UpdateGusts();                       // wind for this step
        const glm::vec3 wind = GetWindVelocity();

        for (RigidBody& body : bodies_) {
            body.ApplyThrusters();                           // 1. thrust (throttle -> force)
            if (body.GetInvMass() > 0.0f) {
                body.ApplyForce(gravity_ * body.GetMass());  // 2. gravity, F = m * g
                if (atmosphere_) {                           // 3. drag, applied at the center of pressure
                    const float rho = AirDensityAt(body.GetPosition().y);
                    const glm::vec3 drag = body.DragForce(rho, dragCoefficient_, wind);
                    body.ApplyForceAtPoint(drag, body.GetCenterOfPressureWorld());
                }
            }
            body.Integrate(kFixedDt);                        // 4. move and rotate
            body.UpdateLegDeployment();                      // 5. latch legs out if descending low
            ResolveGroundContact(body);                      // 6. ground
        }
        ++stepCount_;
    }

    void World::ResolveGroundContact(RigidBody& body) const {
        if (body.GetInvMass() == 0.0f) return;  // static bodies don't respond

        // With no legs out, only the base rim can touch: one pass.
        // With legs out, the base is checked, then each deployed foot, over several passes.
        const bool legsOut = body.HasDeployedLegs();
        const int passes = legsOut ? kLegContactPasses : 1;

        for (int pass = 0; pass < passes; ++pass) {
            ResolveContactPoint(body, -1, restitution_, friction_);
            if (legsOut) {
                for (std::size_t i = 0; i < body.GetLegs().size(); ++i) {
                    if (body.GetLegs()[i].deployed) {
                        ResolveContactPoint(body, static_cast<int>(i), restitution_, friction_);
                    }
                }
            }
        }
    }

}  // namespace rocketphys