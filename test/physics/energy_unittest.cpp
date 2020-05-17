#include <Titan/sim.h> 
#include <chrono>
#include <iostream>

#include <testutil/utils.h>
#include "gtest/gtest.h"

using namespace std;

struct energy_fixture : ::testing::Test {
    double tol = 0.01;
};

TEST_F(energy_fixture, energy_test) {
    titan::Simulation sim;
    sim.createLattice(titan::Vec(0, 0, 5), titan::Vec(4, 4, 4), 20, 20, 20);
    
    sim.setAllSpringConstantValues(100);
    sim.setTimeStep(0.0001);
    sim.setGlobalAcceleration(titan::Vec(0, 0, -9.8));
    sim.defaultRestLengths();

    sim.createPlane(titan::Vec(0, 0, 1), 0);
    sim.start();

    double total_energy = titan::test::energy(sim);
    double avg_energy = total_energy;
    double alpha = 0.7;
    while (sim.time() < 5) {
        sim.wait(0.1);
        avg_energy = (1 - alpha) * titan::test::energy(sim) + alpha * avg_energy;
        EXPECT_NEAR(avg_energy, total_energy, total_energy * tol);

        // cout << "time: " << sim.time() << " total energy: " << total_energy << " energy_diff: " << avg_energy - total_energy << std::endl;

        sim.resume();
    }

    sim.stop();
}
