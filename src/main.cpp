// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>

#include "vec.h"
#include "sim.h"


static Simulation sim;

int main()
{
    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(4, 4, 4), 10, 10, 10);

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

    sim.createPlane(Vec(0, 0, 1), 0);
//    sim.createBall(Vec(0, 0, 0), 2);

#ifdef GRAPHICS
    auto start = std::chrono::high_resolution_clock::now();
    sim.setBreakpoint(5);
    std::cout << "masses: " << sim.masses.size() << " springs: " << sim.springs.size() << std::endl;
    sim.run();
    auto end = std::chrono::high_resolution_clock::now();
#else
    sim.setBreakpoint(0.5);
    sim.run();

    while (sim.time() < 5) {
        sim.printPositions();
        sim.setBreakpoint(sim.time() + 0.5);
        sim.resume();
    }
#endif

    std::cout << (end - start).count() << std::endl;
    return 0;
}
