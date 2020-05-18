#include "gtest/gtest.h"

#include <Titan/sim.h> 

using titan::Vec;

struct rotate_fixture : ::testing::Test {

};

TEST_F(rotate_fixture, rotate_test) {
    titan::Simulation sim;

    titan::Lattice * l2 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    // sim.setViewport(Vec(5, -15, 10), Vec(0, 0, 6), Vec(0, 0, 1));
    // sim.setTimeStep(0.0001);

    sim.setAllSpringConstantValues(1E5);
    l2 -> rotate(Vec(0, 0, 1), -0.78);

    std::cout << sim.masses.size() << " " << sim.springs.size() << std::endl;
    sim.createPlane(Vec(0, 0, 1), 0, 10, 10); // has friction

    sim.setGlobalAcceleration(Vec(0, 0, -9.8));

    sim.start(); // 10 second runtime.

    while (true) {
        sim.pause(sim.time() + 1);
        sim.get(l2);
        l2 -> rotate(Vec(0, 0, 1), 0.5);
        sim.set(l2);

        if (sim.time() > 5.0) {
            sim.stop();
            return;
        }

        sim.resume();
    }
}
