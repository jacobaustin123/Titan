// Include standard headers
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>
#include <chrono>
#include <thread>
#include <cstdlib>

#include "vec.h"
#include "sim.h"


static Simulation sim;

int main()
{
    Lattice * l1 = sim.createLattice(Vec(0, 0, 20), Vec(10, 10, 10), 20, 20, 20);

    sim.setMass(0.1);
    sim.setSpringConstant(10000);
    sim.setMassDeltaT(0.0001);

//    sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)
    sim.createBall(Vec(0, 0, 0), 2);

    std::cout << "running simulation with " << sim.masses.size() << " masses and " << sim.springs.size() << " springs." << std::endl;

    double runtime = 5.0;

    l1 -> masses[0] -> addConstraint(CONSTRAINT_PLANE, Vec(0, 0, 1), 0);

    sim.start();
//    sim.pause(2.0);
//    sim.deleteContainer(l1);
//    sim.resume();

    while (sim.running()) {
        sim.pause(sim.time() + 1.0);

//        sim.createBall(Vec(5 * sin(sim.time()), 5 * cos(sim.time()), 5 * sin(sim.time())), 2);

        sim.get(l1);
////        sim.printPositions();
        l1 -> setKValue(10000 * exp(-sim.time() / 3));
        sim.set(l1);

//        sim.getAll();
//        sim.setSpringConstant(10000 * exp(-sim.time()));
//        sim.setAll();


//        sim.deleteMass(*sim.masses.begin());
//        sim.createMass(Vec(3 * cos(sim.time()), 3 * sin(sim.time()), 15));

//        auto it1 = sim.masses.begin();
//        auto it2 = sim.masses.begin();
//        std::advance(it1, (int) ((double) std::rand() * ((double) (sim.masses.size() - 1) / (double) RAND_MAX)));
//        std::advance(it2, (int) ((double) std::rand() * ((double) (sim.masses.size() - 1) / (double) RAND_MAX)));
//
//        std::cout << sim.masses.size() << " " << std::rand() << " " << (int) ((double) std::rand() * ((double) (sim.masses.size() - 1) / (double) RAND_MAX)) << std::endl;
//
//        sim.createSpring(*it1, *it2);

//        {
//            std::clock_t start;
//            double duration;
//            start = std::clock();
//            sim.createLattice(3 * Vec(cos(sim.time()), 3 * sin(sim.time()), 5), Vec(4, 4, 4), 3, 3, 3);
//            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//            std::cout << "creation time: " << duration << std::endl;
//        }
//
//        if (sim.time() > 5) {
//            std::clock_t start;
//            double duration;
//            start = std::clock();
//            sim.deleteContainer(*sim.objs.begin());
//            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//            std::cout << "deletion time: " << duration << std::endl;
//        }

        if (sim.time() > runtime) {
            sim.stop();
            break;
        }

        sim.resume();
    }

    return 0;
}