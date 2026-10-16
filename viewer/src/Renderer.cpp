#include "Renderer.h"

#include <cstdio>
#include <vector>

#include <glm/gtc/type_ptr.hpp>

namespace {

const char* kVertexSrc = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }
)";

const char* kFragmentSrc = R"(#version 330 core
uniform vec3 uColor;
out vec4 FragColor;
void main() { FragColor = vec4(uColor, 1.0); }
)";

GLuint CompileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error:\n%s\n", log);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

}  // namespace

bool Renderer::Init() {
    // ---- Shader program ----
    GLuint vs = CompileShader(GL_VERTEX_SHADER, kVertexSrc);
    GLuint fs = CompileShader(GL_FRAGMENT_SHADER, kFragmentSrc);
    if (!vs || !fs) return false;

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::fprintf(stderr, "Shader link failed\n");
        return false;
    }
    mvpLoc_ = glGetUniformLocation(program_, "uMVP");
    colorLoc_ = glGetUniformLocation(program_, "uColor");

    // ---- Ground grid: 21 lines each way, 1 m apart, from -10 to +10 at y = 0 ----
    std::vector<glm::vec3> gridVerts;
    for (int i = -10; i <= 10; ++i) {
        gridVerts.push_back({static_cast<float>(i), 0.0f, -10.0f});
        gridVerts.push_back({static_cast<float>(i), 0.0f, 10.0f});
        gridVerts.push_back({-10.0f, 0.0f, static_cast<float>(i)});
        gridVerts.push_back({10.0f, 0.0f, static_cast<float>(i)});
    }
    gridVertexCount_ = static_cast<GLsizei>(gridVerts.size());

    glGenVertexArrays(1, &gridVao_);
    glGenBuffers(1, &gridVbo_);
    glBindVertexArray(gridVao_);
    glBindBuffer(GL_ARRAY_BUFFER, gridVbo_);
    glBufferData(GL_ARRAY_BUFFER, gridVerts.size() * sizeof(glm::vec3), gridVerts.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);

    // ---- Unit cube wireframe (8 corners) + 2 extra vertices for the "forward" line ----
    const float boxVerts[] = {
        -0.5f, -0.5f, -0.5f,  // 0
         0.5f, -0.5f, -0.5f,  // 1
         0.5f,  0.5f, -0.5f,  // 2
        -0.5f,  0.5f, -0.5f,  // 3
        -0.5f, -0.5f,  0.5f,  // 4
         0.5f, -0.5f,  0.5f,  // 5
         0.5f,  0.5f,  0.5f,  // 6
        -0.5f,  0.5f,  0.5f,  // 7
         0.0f,  0.0f,  0.0f,  // 8: center
         0.0f,  0.0f, -1.0f,  // 9: tip of the forward line (-Z, matches the W key direction)
    };
    const unsigned int boxIdx[] = {
        0, 1, 1, 2, 2, 3, 3, 0,   // back face (z = -0.5)
        4, 5, 5, 6, 6, 7, 7, 4,   // front face (z = +0.5)
        0, 4, 1, 5, 2, 6, 3, 7,   // connecting edges
        8, 9,                     // forward line (drawn separately, index offset 24)
    };

    glGenVertexArrays(1, &boxVao_);
    glGenBuffers(1, &boxVbo_);
    glGenBuffers(1, &boxEbo_);
    glBindVertexArray(boxVao_);
    glBindBuffer(GL_ARRAY_BUFFER, boxVbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(boxVerts), boxVerts, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, boxEbo_);  // EBO binding is stored inside the VAO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(boxIdx), boxIdx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);

    glBindVertexArray(0);
    return true;
}

void Renderer::Shutdown() {
    glDeleteVertexArrays(1, &gridVao_);
    glDeleteBuffers(1, &gridVbo_);
    glDeleteVertexArrays(1, &boxVao_);
    glDeleteBuffers(1, &boxVbo_);
    glDeleteBuffers(1, &boxEbo_);
    glDeleteProgram(program_);
}

void Renderer::DrawGrid(const glm::mat4& viewProj) const {
    glUseProgram(program_);
    glUniformMatrix4fv(mvpLoc_, 1, GL_FALSE, glm::value_ptr(viewProj));
    glUniform3f(colorLoc_, 0.35f, 0.40f, 0.48f);
    glBindVertexArray(gridVao_);
    glDrawArrays(GL_LINES, 0, gridVertexCount_);
}

void Renderer::DrawBody(const glm::mat4& viewProj, const glm::mat4& model) const {
    const glm::mat4 mvp = viewProj * model;
    glUseProgram(program_);
    glUniformMatrix4fv(mvpLoc_, 1, GL_FALSE, glm::value_ptr(mvp));
    glBindVertexArray(boxVao_);

    glUniform3f(colorLoc_, 0.95f, 0.95f, 0.95f);
    glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, nullptr);

    glUniform3f(colorLoc_, 1.0f, 0.55f, 0.15f);
    glDrawElements(GL_LINES, 2, GL_UNSIGNED_INT,
                   reinterpret_cast<const void*>(24 * sizeof(unsigned int)));
}
