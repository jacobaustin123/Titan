// Include standard headers
//#include <stdio.h>
//#include <stdlib.h>
//#include <cmath>
//#include <ctime>
//#include <chrono>
//#include <thread>
//#include <cstdlib>

#include "sim.h"

int main()
{
    Simulation sim;

    Lattice * l1 = sim.createLattice(Vec(0, 0, 20), Vec(10, 10, 10), 20, 20, 20);

//    Mass * m1 = sim.createMass(Vec(0, 0, 5));
//    Mass * m2 = sim.createMass(Vec(0, 0, 7));
//    Spring * s1 = sim.createSpring(m1, m2);
//
//    Container * c1 = sim.createContainer();
//    c1 -> add(m1);
//    c1 -> add(m2);
//    c1 -> add(s1);
//
//    c1->setMassValues(0.1);
//    c1->setSpringConstants(10000);
//    c1->setDeltaT(0.0001);

//    sim.createPlane(Vec(0, 0, 1), 0); // add a constraint (can't go below plane z = 0)
        
    sim.createBall(Vec(0, 0, 0), 2);

//    std::cout << "running simulation with " << sim.masses.size() << " masses and " << sim.springs.size() << " springs." << std::endl;

    double runtime = 20.0;

//    l1 -> masses[0] -> addConstraint(DIRECTION, Vec(0, 1, 0), 0);
    sim.start();
//    sim.pause(2.0);
//    sim.deleteContainer(l1);
//    sim.resume();

    while (sim.running()) {
        sim.pause(sim.time() + 1.0);

//        sim.createBall(Vec(5 * sin(sim.time()), 5 * cos(sim.time()), 5 * sin(sim.time())), 2);

//        std::cout << "getting" << std::endl;
//
//        sim.get(l1);
//
//        std::cout << "printing" << std::endl;
//
//        sim.printPositions();
//
//        l1->setSpringConstants(10000 * exp(-sim.time() / 3));
//        sim.set(l1);

//        sim.getAll();
//        sim.setSpringConstants(10000 * exp(-sim.time()));
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

//        std::cout << "starting new iteration" << std::endl;
//
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
//            sim.deleteContainer(sim.getContainerByIndex(0));
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