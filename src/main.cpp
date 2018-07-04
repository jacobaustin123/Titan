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
    sim.createLattice(Vec(0, 0, 20), Vec(6, 6, 6), 5, 5, 5);

    sim.setMass(0.1);
    sim.setSpringConstant(10000);
    sim.setMassDeltaT(0.0001);

    Plane * p = sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)
//    Ball * b = sim.createBall(Vec(0, 0, 0), 5);

    std::cout << "running simulation with " << sim.masses.size() << " masses and " << sim.springs.size() << " springs." << std::endl;

    double runtime = 10.0;

    sim.start();

    while (sim.running()) {
        sim.pause(sim.time() + 1.0);

        std::cout << sim.objs.size() << std::endl;

        {
            std::clock_t start;
            double duration;
            start = std::clock();
            sim.createLattice(3 * Vec(cos(sim.time()), 3 * sin(sim.time()), 5), Vec(4, 4, 4), 3, 3, 3);
            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            std::cout << "creation time: " << duration << std::endl;
        }

        if (sim.time() > 5.0) {
            std::clock_t start;
            double duration;
            start = std::clock();
            sim.deleteContainer(*sim.objs.begin());
            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            std::cout << "deletion time: " << duration << std::endl;
        }

        if (sim.time() > runtime) {
            break;
        } else {
            sim.resume();
        }
    }

    std::cout << "exiting" << std::endl;

    sim.ENDED = true;

    return 0;
}