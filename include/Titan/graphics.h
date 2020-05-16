//
// Created by Jacob Austin on 5/29/18.
//

#ifdef GRAPHICS
#ifndef TITAN_GRAPHICS_H
#define TITAN_GRAPHICS_H

#define GLM_FORCE_PURE

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

#include "vec.h"

namespace titan {

glm::mat4 getProjection(const titan::Vec & camera, const titan::Vec & looks_at, const titan::Vec & up);

} // namespace titan

#endif //TITAN_GRAPHICS_H
#endif