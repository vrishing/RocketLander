#pragma once
// Renderer: owns every OpenGL object. Knows nothing about physics types.

#include <GL/glew.h>
#include <glm/glm.hpp>

class Renderer {
public:
    bool Init();       // compile shaders, upload grid + box geometry. Returns false on failure.
    void Shutdown();   // delete GL objects (call before destroying the GL context)

    void DrawGrid(const glm::mat4& viewProj) const;
    // Draws a unit cube transformed by `model`, plus a line out of its -Z face showing "forward".
    void DrawBody(const glm::mat4& viewProj, const glm::mat4& model) const;

private:
    GLuint program_ = 0;
    GLint mvpLoc_ = -1;
    GLint colorLoc_ = -1;

    GLuint gridVao_ = 0, gridVbo_ = 0;
    GLsizei gridVertexCount_ = 0;

    GLuint boxVao_ = 0, boxVbo_ = 0, boxEbo_ = 0;
};
