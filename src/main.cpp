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

#include "graphics.cpp"
static Simulation sim;

// #cmakedefine SOURCE_DIR "${SOURCE_DIR}"

void translateMass() {
    ContainerObject * o = sim.getObject(0);
    o -> translate(3 * Vec(cos(sim.time()), sin(sim.time()), 0));
}

void doSomething() { // user defined function
    Mass * m1 = sim.getMass(0);

    if (m1->getPosition()[0] < 5)
        m1->setPos(Vec(0, 1, 0)); // no time specified, fulfilled immediately.
}

int main()
{
    Cube * c = sim.createCube(Vec(0, 0, 10), 2.0); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(1000); // set the spring constant for all springs to 10
    c -> setMassValue(1.0); // set all masses to 1.0
    c -> setDeltaTValue(0.00001); // set the dt value for all masses in the cube to 0.00005

    for (Spring * s : c -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }

    Mass * m1 = sim.createMass();
    Mass * m2 = sim.createMass();
    Spring * s1 = sim.createSpring(m1, m2);

    sim.runFunc(translateMass, run_at_time = 0, duration = 0.0001); // schedule function to run at time 0, repeat every 2.5 s

    sim.setBreakpoint(20000); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();


    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)

    sim.runFunc(translateMass, 0, 2.5); // schedule function to run at time 0, repeat every 2.5 s

    sim.setBreakpoint(20000); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();

    return 0;
}
