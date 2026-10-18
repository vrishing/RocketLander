# Python bindings: builds rocketphys_py, a module Python can import.

# Python interpreter and development headers, both needed to compile an extension module.
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 is fetched from GitHub at a pinned version, so no vcpkg feature is needed.
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(pybind11)

# The physics library gets linked into a shared module, so its code must be position independent.
set_target_properties(rocketphys PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(rocketphys_py rocketphys_py.cpp)
target_link_libraries(rocketphys_py PRIVATE rocketphys)

# After each build, copy the module into python/ so scripts there can import it.
add_custom_command(TARGET rocketphys_py POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:rocketphys_py>
            ${CMAKE_SOURCE_DIR}/python/)