#pragma once
// World: owns bodies, applies gravity, drag, wind, and the ground. Steps time in FIXED increments.

#include <cstddef>
#include <cstdint>
#include <random>
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

        // Air. Off by default (vacuum). seaLevelDensity is in kg/m^3.
        void SetAtmosphere(bool enabled, float seaLevelDensity = 1.225f);
        bool HasAtmosphere() const { return atmosphere_; }

        // Drag coefficient for the body. 0.8 is a reasonable blunt-cylinder value.
        void SetDragCoefficient(float cd) { dragCoefficient_ = (cd < 0.0f) ? 0.0f : cd; }
        float GetDragCoefficient() const { return dragCoefficient_; }

        // Air density at a height in meters: rho(h) = rho0 * exp(-h / 8500).
        float AirDensityAt(float height) const;

        // ---------- Wind ----------
        // Wind is air motion. It only has an effect while the atmosphere is on, because it acts through drag.
        void SetBaseWind(const glm::vec3& wind) { baseWind_ = wind; }
        const glm::vec3& GetBaseWind() const { return baseWind_; }

        // Gusts: every gustInterval seconds a new random horizontal gust target is chosen, with
        // speed up to maxGustSpeed. The gust eases toward its target, so it never jumps.
        // The same seed always gives the same gust sequence.
        void SetGusts(bool enabled, float maxGustSpeed, float gustInterval, std::uint32_t seed);

        // Total wind right now: base wind plus the current gust.
        glm::vec3 GetWindVelocity() const { return baseWind_ + gustNow_; }

        // Advance the whole world by exactly kFixedDt.
        void Step();

        std::uint64_t GetStepCount() const { return stepCount_; }
        double GetTime() const { return static_cast<double>(stepCount_) * kFixedDt; }

    private:
        void ResolveGroundContact(RigidBody& body) const;
        void UpdateGusts();

        std::vector<RigidBody> bodies_;
        glm::vec3 gravity_{ 0.0f, -9.81f, 0.0f };
        float restitution_ = 0.3f;
        float friction_ = 0.6f;

        bool atmosphere_ = false;
        float seaLevelDensity_ = 1.225f;
        float dragCoefficient_ = 0.8f;

        glm::vec3 baseWind_{ 0.0f };
        bool gustsEnabled_ = false;
        float maxGustSpeed_ = 0.0f;
        float gustInterval_ = 5.0f;
        double nextGustTime_ = 0.0;
        glm::vec3 gustTarget_{ 0.0f };
        glm::vec3 gustNow_{ 0.0f };
        std::mt19937 rng_{ 0 };

        std::uint64_t stepCount_ = 0;
    };

}  // namespace rocketphys