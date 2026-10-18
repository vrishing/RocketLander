#include "rocketphys/RigidBody.h"

#include <cmath>
#include <algorithm>

namespace rocketphys {

    void RigidBody::SetMass(float mass) {
        mass_ = mass;
        invMass_ = (mass > 0.0f) ? 1.0f / mass : 0.0f;
    }

    void RigidBody::SetBoxInertia(float width, float height, float depth) {
        invInertiaBody_ = glm::mat3(0.0f);
        if (invMass_ == 0.0f) return;

        const float k = mass_ / 12.0f;
        const float ixx = k * (height * height + depth * depth);
        const float iyy = k * (width * width + depth * depth);
        const float izz = k * (width * width + height * height);

        invInertiaBody_[0][0] = 1.0f / ixx;
        invInertiaBody_[1][1] = 1.0f / iyy;
        invInertiaBody_[2][2] = 1.0f / izz;
    }

    void RigidBody::SetCylinderInertia(float radius, float height) {
        // Shape is recorded before the static-body return, so contact code always has it.
        radius_ = radius;
        halfHeight_ = height * 0.5f;

        invInertiaBody_ = glm::mat3(0.0f);
        if (invMass_ == 0.0f) return;

        const float m = mass_;
        const float iyy = 0.5f * m * radius * radius;                                   // long axis (Y)
        const float ixx = (1.0f / 12.0f) * m * (3.0f * radius * radius + height * height);
        const float izz = ixx;                                                          // sideways axes

        invInertiaBody_[0][0] = 1.0f / ixx;
        invInertiaBody_[1][1] = 1.0f / iyy;
        invInertiaBody_[2][2] = 1.0f / izz;
    }

    glm::mat3 RigidBody::GetInvInertiaBody() const {
        return invInertiaBody_;
    }

    void RigidBody::SetDamping(float linear, float angular) {
        linearDamping_ = linear;
        angularDamping_ = angular;
    }

    void RigidBody::ApplyForce(const glm::vec3& worldForce) { force_ += worldForce; }

    void RigidBody::ApplyTorque(const glm::vec3& worldTorque) { torque_ += worldTorque; }

    void RigidBody::ApplyForceAtPoint(const glm::vec3& worldForce, const glm::vec3& worldPoint) {
        force_ += worldForce;
        torque_ += glm::cross(worldPoint - position_, worldForce);
    }

    void RigidBody::ZeroMotion() {
        velocity_ = glm::vec3(0.0f);
        angularVelocity_ = glm::vec3(0.0f);
        force_ = glm::vec3(0.0f);
        torque_ = glm::vec3(0.0f);
    }

    glm::vec3 RigidBody::GetLowestPoint() const {
        // World direction of the cylinder's long axis.
        const glm::vec3 axis = GetRotationMatrix() * glm::vec3(0.0f, 1.0f, 0.0f);
        const float a = axis.y;  // 1 = upright, 0 = lying flat

        // Direction along the bottom disk that points most steeply down.
        // Its length is sqrt(1 - a^2), and it is exactly zero when upright.
        const glm::vec3 up(0.0f, 1.0f, 0.0f);
        const glm::vec3 perp = up - a * axis;
        const float perpLength = glm::length(perp);
        glm::vec3 rimDown(0.0f);
        if (perpLength > 1e-6f) rimDown = -perp / perpLength;

        // The end of the axis that is lower in the world.
        const float endSign = (a > 0.0f) ? -1.0f : 1.0f;

        return position_ + endSign * halfHeight_ * axis + radius_ * rimDown;
    }

    float RigidBody::GetTiltDegrees() const {
        const glm::vec3 up = GetRotationMatrix() * glm::vec3(0.0f, 1.0f, 0.0f);
        return glm::degrees(std::acos(glm::clamp(up.y, -1.0f, 1.0f)));
    }

    void RigidBody::RecordTouchdown(float speedMps, float tiltDeg) {
        if (landingStatus_ != LandingStatus::Flying) return;  // only the first contact counts

        touchdownSpeed_ = speedMps;
        touchdownTiltDeg_ = tiltDeg;
        landingStatus_ = ClassifyTouchdown(touchdownSpeed_, touchdownTiltDeg_);
    }

    void RigidBody::ResetLandingStatus() {
        landingStatus_ = LandingStatus::Flying;
        touchdownSpeed_ = 0.0f;
        touchdownTiltDeg_ = 0.0f;
    }

    std::size_t RigidBody::AddThruster(const Thruster& t) {
        thrusters_.push_back(t);
        return thrusters_.size() - 1;
    }

    void RigidBody::SetThrottle(std::size_t index, float value) {
        Thruster& t = thrusters_.at(index);  // throws std::out_of_range on a bad index
        t.throttle = (value < 0.0f) ? 0.0f : (value > 1.0f) ? 1.0f : value;
    }

    void RigidBody::ApplyThrusters() {
        const glm::mat3 R = GetRotationMatrix();  // computed once, not per thruster
        for (const Thruster& t : thrusters_) {
            const glm::vec3 worldDirection = R * t.localDirection;
            const glm::vec3 worldForce = worldDirection * t.maxForce * t.throttle;
            const glm::vec3 worldPoint = position_ + R * t.localPosition;
            ApplyForceAtPoint(worldForce, worldPoint);
        }
    }

    void RigidBody::Integrate(float dt) {
        if (invMass_ == 0.0f) {  // static body
            force_ = glm::vec3(0.0f);
            torque_ = glm::vec3(0.0f);
            return;
        }

        // Linear: semi-implicit Euler (velocity first, then position).
        velocity_ += force_ * invMass_ * dt;
        velocity_ *= std::pow(linearDamping_, dt * 60.0f);
        position_ += velocity_ * dt;

        // Angular: world-space inverse inertia is R * Ibody^-1 * R^T.
        const glm::mat3 R = glm::mat3_cast(orientation_);
        const glm::mat3 invInertiaWorld = R * invInertiaBody_ * glm::transpose(R);

        angularVelocity_ += invInertiaWorld * torque_ * dt;
        angularVelocity_ *= std::pow(angularDamping_, dt * 60.0f);

        // dq/dt = 0.5 * (0, w) * q, then renormalize.
        const glm::quat omegaQuat(0.0f, angularVelocity_.x, angularVelocity_.y, angularVelocity_.z);
        orientation_ += (omegaQuat * orientation_) * (0.5f * dt);
        orientation_ = glm::normalize(orientation_);

        force_ = glm::vec3(0.0f);
        torque_ = glm::vec3(0.0f);
    }
    namespace {
        constexpr float kPi = 3.14159265f;
    }  // namespace

    glm::vec3 RigidBody::DragForce(float airDensity, float dragCoefficient,
        const glm::vec3& airVelocity) const {
        // Drag depends on the velocity relative to the air, not to the ground.
        const glm::vec3 relV = velocity_ - airVelocity;
        const float speed = glm::length(relV);
        if (speed < 1e-6f || airDensity <= 0.0f || dragCoefficient <= 0.0f) {
            return glm::vec3(0.0f);
        }

        // Angle between the relative velocity and the long axis, using |cos| since
        // drag does not care which end leads.
        const glm::vec3 up = GetRotationMatrix() * glm::vec3(0.0f, 1.0f, 0.0f);
        const float cosT = std::fabs(glm::dot(relV, up)) / speed;
        const float sinT = std::sqrt(std::max(0.0f, 1.0f - cosT * cosT));

        // Reference area blends between the end disk (pi r^2) and the side wall (2 r h).
        const float height = 2.0f * halfHeight_;
        const float area = kPi * radius_ * radius_ * cosT + 2.0f * radius_ * height * sinT;

        // F = -1/2 * rho * Cd * A * |v_rel| * v_rel. The minus sign opposes the relative motion.
        return -0.5f * airDensity * dragCoefficient * area * speed * relV;
    }
    glm::vec3 RigidBody::GetCenterOfPressureWorld() const {
        return position_ + GetRotationMatrix() * localCenterOfPressure_;
    }
    // ---------- Legs ----------

    std::size_t RigidBody::AddLeg(const Leg& leg) {
        legs_.push_back(leg);
        return legs_.size() - 1;
    }

    bool RigidBody::HasDeployedLegs() const {
        for (const Leg& leg : legs_) {
            if (leg.deployed) return true;
        }
        return false;
    }

    glm::vec3 RigidBody::GetLegFootWorld(std::size_t index) const {
        return position_ + GetRotationMatrix() * legs_.at(index).localFoot;
    }

    void RigidBody::UpdateLegDeployment() {
        if (legs_.empty() || HasDeployedLegs()) return;  // latched: stays out until ResetLegs()

        const float baseAltitude = position_.y - halfHeight_;  // bottom of the body above the floor
        const bool descendingFast = velocity_.y < -kLegDeploySpeed;
        if (baseAltitude < kLegDeployAltitude && descendingFast) {
            for (Leg& leg : legs_) leg.deployed = true;
        }
    }

    void RigidBody::ResetLegs() {
        for (Leg& leg : legs_) leg.deployed = false;
    }

}  // namespace rocketphys