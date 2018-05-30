// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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

#include "graphics.cpp"

// #cmakedefine SOURCE_DIR "${SOURCE_DIR}"

static Simulation sim;

void translateMass() {
    ContainerObject * o = sim.getObject(0);
    o->translate(3 * Vec(cos(sim.time()), sin(sim.time()), 0));
}

int main()
{
    Cube * c = sim.createCube(Vec(0, 0, 10), 2.0); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(1000); // set the spring constant for all springs to 10
    c -> setMassValue(1.0); // set all masses to 1.0
    c -> setDeltaTValue(0.00005); // set the dt value for all masses in the cube to 0.00005

    for (Spring * s : c -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }

    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)

    sim.runFunc(translateMass, 500); // schedule function to run at time 500
    sim.runFunc(translateMass, 1000); // schedule function to run at time 1000;
    sim.runFunc(translateMass, 1500);
    sim.runFunc(translateMass, 2000);
    sim.runFunc(translateMass, 2500);
    sim.runFunc(translateMass, 3000);

    sim.setBreakpoint(20000); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();


    return 0;
}
