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
    Lattice * l1 = sim.createLattice(Vec(0, 0, 20), Vec(15, 15, 5), 2, 2, 2);

    sim.setMass(0.1);
    sim.setSpringConstant(10000);
    sim.setMassDeltaT(0.0001);

//    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)
    Ball * b = sim.createBall(Vec(0, 0, 0), 5);

    std::cout << "running simulation with " << sim.masses.size() << " masses and " << sim.springs.size() << " springs." << std::endl;

    std::clock_t start;
    double duration;
    start = std::clock();

    double runtime = 10.0;

#ifdef GRAPHICS
    sim.setBreakpoint(runtime); // set breakpoint (could be end of program or just time to check for updates)
    sim.start();

    while (sim.running()) {
        std::cout << sim.time() << std::endl;
        sim.pause(sim.time() + 1.0);
        std::cout << sim.running() << std::endl;
        sim.createMass(Vec(0, 0, 5));
//        sim.createLattice(Vec(5 * cos(sim.time()), 5 * sin(sim.time()), 20), Vec(5, 5, 5), 5, 5, 5);
        sim.resume();
    }
#else
    sim.setBreakpoint(runtime); // set breakpoint (could be end of program or just time to check for updates)
    sim.start();

    while (sim.running()) {
        std::cout << sim.time() << std::endl;
        sim.printPositions();
        sim.pause(sim.time() + 1.0);
        sim.createMass(Vec(0, 0, 1));
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        sim.resume();
    }
#endif

    std::cout << "This is printing while the program is still running!" << std::endl;

    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"wall time for " << runtime << " second start with " << sim.masses.size()
             << " masses and " << sim.springs.size() << " springs is " << duration << "!" << std::endl;


    std::this_thread::sleep_for(20s);

    return 0;
}