cmake_minimum_required(VERSION 3.20)
project(RocketLander LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_TESTS          "Build unit tests"                ON)
option(BUILD_VIEWER         "Build the SDL/OpenGL viewer"     ON)
option(BUILD_PYTHON_BINDINGS "Build the Python module"        ON)

# The physics library never depends on SDL/OpenGL/ImGui or Python.
add_subdirectory(physics)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BUILD_VIEWER)
    add_subdirectory(viewer)
endif()

if(BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()