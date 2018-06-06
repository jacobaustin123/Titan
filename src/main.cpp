// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "vec.h"
#include "sim.h"


static Simulation sim;

void translateMass() {
    ContainerObject * o = sim.getObject(0);
    o->translate(Vec(cos(sim.time()), sin(sim.time()), 0));
}

int main()
{
//    Simulation sim; // initialize simulation object

    Cube * c = sim.createCube(Vec(2, -2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(1000); // set the spring constant for all springs to 10
    c -> setMassValue(1.0); // set all masses to 2.0
    c -> setDeltaTValue(0.0001); // set the dt value for all masses in the cube to 0.00001

    for (Spring * s : c -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }
    
    Cube * c1 = sim.createCube(Vec(-2, 2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0
    c1 -> setKValue(1000); // set the spring constant for all springs to 10
    c1 -> setMassValue(1.0); // set all masses to 2.0
    c1 -> setDeltaTValue(0.0001); // set the dt value for all masses in the cube to 0.00001

    for (Spring * s : c1 -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }
    
    sim.createPlane(Vec(0, 0, 1), 0);


#ifdef GRAPHICS
    sim.setBreakpoint(20);
    sim.runFunc(translateMass, 1.0);

    sim.run();
#else
    sim.setBreakpoint(0.5);
    sim.runFunc(translateMass, 1.0);
    sim.run();

    while (sim.time() < 10) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.5);
        sim.resume();
    }
#endif

    return 0;
}
