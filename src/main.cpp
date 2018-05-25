// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "vec.h"
#include "sim.h"

int main()
{
    Simulation sim; // initialize simulation object

    Cube * c = sim.createCube(Vec(0, 0, 5), 0.5); // create Cube object centered at (0, 0, 10) with side length 2.0
    c -> setKValue(1000); // set the spring constant for all springs to 10
    c -> setMassValue(1.0); // set all masses to 2.0
    c -> setDeltaTValue(0.00001); // set the dt value for all masses in the cube to 0.00001

    for (Spring * s : c -> springs) {
        s -> setRestLength((s -> _right->getPosition() - s -> _left->getPosition()).norm());
    }
//    sim.createPlane(Vec(0, 0, 1), 0);

    sim.printPositions();
    
    sim.setBreakpoint(0.01);
    sim.run();

    while ( sim.time() < 10.0 ) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.5);
        sim.resume();
    }

    return 0;
}
