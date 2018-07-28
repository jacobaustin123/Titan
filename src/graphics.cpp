#ifdef GRAPHICS

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// Include shaders and support files
#include "shader.h"
#include "vec.h"
#include "sim.h"

#include "graphics.h"

glm::mat4 getProjection(const Vec & camera, const Vec & looks_at, const Vec & up) {
    // Projection matrix : 45∞ Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
    glm::mat4 Projection = glm::perspective(glm::radians(45.0f), 4.0f / 3.0f, 0.1f, 100.0f);
    // Or, for an ortho camera :
    //glm::mat4 Projection = glm::ortho(-10.0f,10.0f,-10.0f,10.0f,0.0f,100.0f); // In world coordinates

    // Camera matrix
    glm::mat4 View = glm::lookAt(
            glm::vec3(camera[0], camera[1], camera[2]), // Camera is at (4,3,3), in World Space
            glm::vec3(looks_at[0], looks_at[1], looks_at[2]), // and looks at the origin
            glm::vec3(up[0], up[1], up[2])  // Head is up (set to 0,-1,0 to look upside-down)
    );
    // Model matrix : an identity matrix (model will be at the origin)
    glm::mat4 Model = glm::mat4(1.0f);
    // Our ModelViewProjection : multiplication of our 3 matrices
    glm::mat4 MVP = Projection * View * Model; // Remember, matrix multiplication is the other way around

    return MVP;
}

#endif