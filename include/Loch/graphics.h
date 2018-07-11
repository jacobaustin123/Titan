//
// Created by Jacob Austin on 5/29/18.
//

#ifndef LOCH_GRAPHICS_H
#define LOCH_GRAPHICS_H

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
GLFWwindow * createGLFWWindow();
glm::mat4 getProjection();

#endif //LOCH_GRAPHICS_H
