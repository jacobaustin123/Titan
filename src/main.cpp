// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// Include shaders and support files
#include "common/shader.h"
#include "vec.h"
#include "sim.h"

#include "graphics.h"

static Simulation sim;

int main()
{
    Cube * c = sim.createCube(Vec(2, -2, 8), 2.0); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(500); // set the spring constant for all springs to 10
    c -> setMassValue(1.0); // set all masses to 1.0
    c -> setDeltaTValue(0.0001); // set the dt value for all masses in the cube to 0.00005

    for (Spring * s : c -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }

    Cube * c2 = sim.createCube(Vec(-2, 2, 8), 2.0); // create Cube object centered at (0, 0, 10) with side length 2.0
    c2 -> setKValue(500); // set the spring constant for all springs to 10
    c2 -> setMassValue(1.0); // set all masses to 1.0
    c2 -> setDeltaTValue(0.0001); // set the dt value for all masses in the cube to 0.00005

    for (Spring * s : c2 -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }

    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)

    sim.setBreakpoint(100); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();

    return 0;
}