#include "rocketphys/Rocket.h"

#include "rocketphys/Thruster.h"

namespace rocketphys {

    RigidBody MakeRocketBody() {
        RigidBody rocket;
        rocket.SetMass(1000.0f);
        rocket.SetCylinderInertia(0.5f, 8.0f);  // must come after SetMass
        rocket.SetDamping(1.0f, 1.0f);          // no air in the simulation
        rocket.SetHalfHeight(4.0f);             // center to lowest point

        // Order here MUST match the k... index constants in Rocket.h.
        // Thruster(localPosition, localDirection, maxForce)
        rocket.AddThruster(Thruster({ 0.0f, -4.0f,  0.0f }, { 0.0f, 1.0f,  0.0f }, 25000.0f));  // main

        rocket.AddThruster(Thruster({ 0.0f,  3.0f,  0.5f }, { 0.0f, 0.0f, -1.0f }, 1000.0f));  // pitch A top
        rocket.AddThruster(Thruster({ 0.0f, -3.0f, -0.5f }, { 0.0f, 0.0f,  1.0f }, 1000.0f));  // pitch A bottom
        rocket.AddThruster(Thruster({ 0.0f,  3.0f, -0.5f }, { 0.0f, 0.0f,  1.0f }, 1000.0f));  // pitch B top
        rocket.AddThruster(Thruster({ 0.0f, -3.0f,  0.5f }, { 0.0f, 0.0f, -1.0f }, 1000.0f));  // pitch B bottom

        rocket.AddThruster(Thruster({ 0.5f,  3.0f,  0.0f }, { -1.0f, 0.0f,  0.0f }, 1000.0f));  // roll A top
        rocket.AddThruster(Thruster({ -0.5f, -3.0f,  0.0f }, { 1.0f, 0.0f,  0.0f }, 1000.0f));  // roll A bottom
        rocket.AddThruster(Thruster({ -0.5f,  3.0f,  0.0f }, { 1.0f, 0.0f,  0.0f }, 1000.0f));  // roll B top
        rocket.AddThruster(Thruster({ 0.5f, -3.0f,  0.0f }, { -1.0f, 0.0f,  0.0f }, 1000.0f));  // roll B bottom

        return rocket;
    }

}  // namespace rocketphys