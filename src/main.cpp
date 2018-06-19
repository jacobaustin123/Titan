// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "vec.h"
#include "sim.h"


static Simulation sim;

int main()
{
    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(4, 4, 2), 3, 3, 2);

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

//    sim.createPlane(Vec(0, 0, 1), 0);
    sim.createBall(Vec(0, 0, 0), 2);

#ifdef GRAPHICS
    sim.setBreakpoint(40);
    sim.run();
#else
    sim.setBreakpoint(0.5);
    sim.run();

    while (sim.time() < 5) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.5);
        sim.resume();
    }
#endif

    return 0;
}
