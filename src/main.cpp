// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "vec.h"
#include "sim.h"


static Simulation sim;

//void translateMass() {
//    ContainerObject * o = sim.getObject(0);
//    o->translate(Vec(cos(sim.time()), sin(sim.time()), 0));
//}

int main()
{
//    Cube * c1 = sim.createCube(Vec(2, -2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0
//    Cube * c2 = sim.createCube(Vec(-2, 2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0
//    Cube * c3 = sim.createCube(Vec(2, 2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0
//    Cube * c4 = sim.createCube(Vec(-2, -2, 5), 2); // create Cube object centered at (0, 0, 10) with side length 2.0

    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(4, 4, 2), 2, 2, 2);

    sim.printPositions();

    std::cout << l1 -> springs.size() << std::endl;

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

    sim.createPlane(Vec(0, 0, 1), 0);

#ifdef GRAPHICS
    sim.setBreakpoint(40);
    sim.run();
#else
    sim.setBreakpoint(0.1);
    sim.run();

    while (sim.time() < 1) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.1);
        sim.resume();
    }
#endif

    return 0;
}
