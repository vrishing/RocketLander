find_package(glm CONFIG REQUIRED)

add_library(rocketphys STATIC
    src/RigidBody.cpp
    src/World.cpp
 "include/rocketphys/Thruster.h" "include/rocketphys/Rocket.h" "src/Rocket.cpp" "include/rocketphys/Landing.h" "include/rocketphys/Leg.h")

# PUBLIC: anything linking rocketphys can #include <rocketphys/...>
target_include_directories(rocketphys PUBLIC include)

# PUBLIC: our headers expose glm types, so users need glm too.
target_link_libraries(rocketphys PUBLIC glm::glm)

if(MSVC)
    target_compile_options(rocketphys PRIVATE /W4)
else()
    target_compile_options(rocketphys PRIVATE -Wall -Wextra)
endif()
