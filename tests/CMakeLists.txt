find_package(GTest CONFIG REQUIRED)
include(GoogleTest)

add_executable(physics_tests test_physics.cpp test_env.cpp)
target_link_libraries(physics_tests PRIVATE rocketphys GTest::gtest GTest::gtest_main)

# Registers every TEST() with CTest so Visual Studio's Test Explorer sees them.
gtest_discover_tests(physics_tests)
