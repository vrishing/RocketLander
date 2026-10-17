find_package(SDL2 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(viewer
    src/main.cpp
    src/Renderer.cpp
    
)

# Keeps main() as a plain main() instead of SDL's main wrapper.
target_compile_definitions(viewer PRIVATE SDL_MAIN_HANDLED)

target_link_libraries(viewer PRIVATE
    rocketphys
    SDL2::SDL2
    GLEW::GLEW
    OpenGL::GL
)

if(MSVC)
    target_compile_options(viewer PRIVATE /W4)
else()
    target_compile_options(viewer PRIVATE -Wall -Wextra)
endif()
