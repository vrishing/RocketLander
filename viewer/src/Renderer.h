#pragma once
// Renderer: owns every OpenGL object. Knows nothing about physics types.

#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>

class Renderer {
public:
    bool Init();       // compile shaders, upload grid + box + line geometry
    void Shutdown();   // delete GL objects (call before destroying the GL context)

    void DrawGrid(const glm::mat4& viewProj) const;
    // Draws a unit cube transformed by `model`, plus a line out of its -Z face showing "forward".
    void DrawBody(const glm::mat4& viewProj, const glm::mat4& model) const;
    // Draws line segments from consecutive pairs: (p0,p1), (p2,p3), ...
    void DrawLines(const glm::mat4& viewProj, const std::vector<glm::vec3>& points,
        const glm::vec3& color) const;

private:
    GLuint program_ = 0;
    GLint mvpLoc_ = -1;
    GLint colorLoc_ = -1;

    GLuint gridVao_ = 0, gridVbo_ = 0;
    GLsizei gridVertexCount_ = 0;

    GLuint boxVao_ = 0, boxVbo_ = 0, boxEbo_ = 0;

    // Dynamic buffer for line segments, rewritten every frame.
    GLuint lineVao_ = 0, lineVbo_ = 0;
};