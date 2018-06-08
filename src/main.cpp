// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include "vec.h"
#include "sim.h"


static Simulation sim;

int main()
{
    Lattice * l1 = sim.createLattice(Vec(0, 0, 20), Vec(15, 15, 15), 10, 10, 10);

    sim.setSpringConstant(10000);
    sim.setMassDeltaT(0.0001);

    sim.createPlane(Vec(0, 0, 1), 0);

    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)

#ifdef GRAPHICS
        sim.setBreakpoint(100); // set breakpoint (could be end of program or just time to check for updates)
        sim.run();
#else
    sim.setBreakpoint(0.1); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();

    while (sim.time() < 5) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.5);
        sim.resume();
    }

#endif

    return 0;
}