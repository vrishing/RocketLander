// Python bindings for the physics library. This file only translates types between C++ and
// Python. It contains no physics logic.

#include <array>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rocketphys/Baseline.h"
#include "rocketphys/Landing.h"
#include "rocketphys/LandingEnv.h"

namespace py = pybind11;
using namespace rocketphys;

PYBIND11_MODULE(rocketphys_py, m) {
    m.doc() = "Rocket landing physics and RL environment (C++ core, Python interface).";

    // Landing outcome as a Python enum, so code can read: if result.status == LandingStatus.Landed.
    py::enum_<LandingStatus>(m, "LandingStatus")
        .value("Flying", LandingStatus::Flying)
        .value("Landed", LandingStatus::Landed)
        .value("Crashed", LandingStatus::Crashed);

    // Reward weights. Python attribute names use snake_case, matching Python style.
    py::class_<RewardWeights>(m, "RewardWeights")
        .def(py::init<>())
        .def_readwrite("step_cost", &RewardWeights::stepCost)
        .def_readwrite("tilt_cost", &RewardWeights::tiltCost)
        .def_readwrite("spin_cost", &RewardWeights::spinCost)
        .def_readwrite("discount", &RewardWeights::discount)
        .def_readwrite("altitude_shaping", &RewardWeights::altitudeShaping)
        .def_readwrite("timeout_penalty", &RewardWeights::timeoutPenalty)
        .def_readwrite("landed_bonus", &RewardWeights::landedBonus)
        .def_readwrite("crash_penalty", &RewardWeights::crashPenalty)
        .def_readwrite("touchdown_speed_cost", &RewardWeights::touchdownSpeedCost)
        .def_readwrite("touchdown_tilt_cost", &RewardWeights::touchdownTiltCost);

    // Environment configuration.
    py::class_<EnvConfig>(m, "EnvConfig")
        .def(py::init<>())
        .def_readwrite("action_repeat", &EnvConfig::actionRepeat)
        .def_readwrite("max_episode_seconds", &EnvConfig::maxEpisodeSeconds)
        .def_readwrite("min_start_height", &EnvConfig::minStartHeight)
        .def_readwrite("max_start_height", &EnvConfig::maxStartHeight)
        .def_readwrite("max_start_speed", &EnvConfig::maxStartSpeed)
        .def_readwrite("difficulty", &EnvConfig::difficulty)
        .def_readwrite("atmosphere", &EnvConfig::atmosphere)
        .def_readwrite("max_wind_speed", &EnvConfig::maxWindSpeed)
        .def_readwrite("gust_interval", &EnvConfig::gustInterval)
        .def_readwrite("wind_sensor", &EnvConfig::windSensor)
        .def_readwrite("reward", &EnvConfig::reward)
        // glm::vec3 is unknown to pybind11, so it crosses the boundary as a 3-element list.
        .def_property("base_wind",
            [](const EnvConfig& c) {
                return std::array<float, 3>{c.baseWind.x, c.baseWind.y, c.baseWind.z};
            },
            [](EnvConfig& c, const std::array<float, 3>& v) {
                c.baseWind = { v[0], v[1], v[2] };
            });

    // The environment. Actions are 3-element lists, observations are lists of floats.
    py::class_<LandingEnv>(m, "LandingEnv")
        .def(py::init<>())
        .def(py::init<const EnvConfig&>(), py::arg("config"))
        .def("reset", &LandingEnv::Reset, py::arg("seed"))
        .def("step", &LandingEnv::Step, py::arg("action"))
        .def("observation", &LandingEnv::GetObservation)
        .def("set_difficulty", &LandingEnv::SetDifficulty, py::arg("difficulty"))
        .def_property_readonly("observation_size", &LandingEnv::ObservationSize)
        .def_property_readonly("sim_time", &LandingEnv::SimTime)
        .def_property_readonly("height", [](const LandingEnv& e) {
        return e.Rocket().GetLowestPoint().y;
            })
        .def_property_readonly("tilt_deg", [](const LandingEnv& e) {
        return e.Rocket().GetTiltDegrees();
            });

    // What step() returns.
    py::class_<StepResult>(m, "StepResult")
        .def_readonly("observation", &StepResult::observation)
        .def_readonly("reward", &StepResult::reward)
        .def_readonly("terminated", &StepResult::terminated)
        .def_readonly("truncated", &StepResult::truncated)
        .def_readonly("status", &StepResult::status);

    // The hand-designed reference controller. It returns the same action format as the policy.
    py::class_<LandingBaseline>(m, "LandingBaseline")
        .def(py::init<>())
        .def("act", [](const LandingBaseline& pilot, const LandingEnv& env) {
        const Action a = pilot.Compute(env.Rocket());
        return std::vector<float>{a[0], a[1], a[2]};
            }, py::arg("env"));
}