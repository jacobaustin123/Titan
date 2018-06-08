// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
using namespace std::literals::chrono_literals;

#include "vec.h"
#include "sim.h"


static Simulation sim;

int main()
{
    Lattice * l1 = sim.createLattice(Vec(0, 0, 20), Vec(15, 15, 15), 10, 10, 10);

//    Mass * m1 = sim.createMass(Vec(0, 0, 20));
//    Mass * m2 = sim.createMass(Vec(0, 10, 20));
//    m1 -> makeFixed();
//    Spring * s1 = sim.createSpring(m1, m2);
//    s1 -> defaultLength();
//
//    sim.setSpringConstant(10);

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

    sim.createPlane(Vec(0, 0, 1), 0);

    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)

    std::cout << "running simulation with " << sim.masses.size() << " masses and " << sim.springs.size() << " springs." << std::endl;

#ifndef GRAPHICS
    std::clock_t start;
    double duration;
    start = std::clock();
#endif

    double runtime = 10.0;

#ifdef GRAPHICS
        sim.setBreakpoint(runtime); // set breakpoint (could be end of program or just time to check for updates)
        sim.run();
#else
    sim.setBreakpoint(runtime); // set breakpoint (could be end of program or just time to check for updates)
    sim.run();
//
//    while (sim.time() < 5) {
//        sim.printPositions();
//        sim.setBreakpoint(sim.time() + 0.5);
//        sim.resume();
//    }

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"wall time for " << runtime << " second run with " << sim.masses.size()
             << " masses and " << sim.springs.size() << " springs is " << duration << "!" << std::endl;


    std::this_thread::sleep_for(5s);
#endif

    return 0;
}