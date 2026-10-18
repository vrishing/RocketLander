#pragma once
// World: owns bodies, applies gravity, steps time in FIXED increments, handles the ground.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "rocketphys/RigidBody.h"

namespace rocketphys {

    class World {
    public:
        // Every Step() advances exactly this much. Never pass a variable dt to physics.
        static constexpr float kFixedDt = 1.0f / 120.0f;

        // Returns the index of the new body. Use GetBody(index) to keep configuring it.
        std::size_t AddBody(const RigidBody& body);
        RigidBody& GetBody(std::size_t index) { return bodies_.at(index); }
        const RigidBody& GetBody(std::size_t index) const { return bodies_.at(index); }
        std::size_t BodyCount() const { return bodies_.size(); }

        void SetGravity(const glm::vec3& g) { gravity_ = g; }
        void SetRestitution(float e) { restitution_ = e; }

        // Coulomb friction coefficient at the ground. 0 = frictionless, 0.6 = rubber on concrete.
        void SetFriction(float mu) { friction_ = (mu < 0.0f) ? 0.0f : mu; }
        float GetFriction() const { return friction_; }

        // Advance the whole world by exactly kFixedDt.
        void Step();

        std::uint64_t GetStepCount() const { return stepCount_; }
        double GetTime() const { return static_cast<double>(stepCount_) * kFixedDt; }

    private:
        void ResolveGroundContact(RigidBody& body) const;

        std::vector<RigidBody> bodies_;
        glm::vec3 gravity_{ 0.0f, -9.81f, 0.0f };
        float restitution_ = 0.3f;
        float friction_ = 0.6f;
        std::uint64_t stepCount_ = 0;
    };

}  // namespace rocketphys