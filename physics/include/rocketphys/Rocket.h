#pragma once
#include <cstddef>

#include "rocketphys/RigidBody.h"

namespace rocketphys {

	// Thruster indices, in the order MakeRocketBody() adds them.
	inline constexpr std::size_t kMainEngine = 0;
	inline constexpr std::size_t kPitchATop = 1;
	inline constexpr std::size_t kPitchABottom = 2;
	inline constexpr std::size_t kPitchBTop = 3;
	inline constexpr std::size_t kPitchBBottom = 4;
	inline constexpr std::size_t kRollATop = 5;
	inline constexpr std::size_t kRollABottom = 6;
	inline constexpr std::size_t kRollBTop = 7;
	inline constexpr std::size_t kRollBBottom = 8;

	// 1000 kg cylinder, radius 0.5 m, height 8 m, centered on the origin.
	RigidBody MakeRocketBody();

}  // namespace rocketphys