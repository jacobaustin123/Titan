#include <Titan/sim.h> 
#include <chrono>
#include <iostream>

#include "gtest/gtest.h"

using namespace std;

double energy(titan::Simulation & sim) {
    double potential_g = 0;
    double potential_s = 0;
    double kinetic = 0;
    sim.getAll();

    for (titan::Mass * m : sim.masses) {
        potential_g += 9.8 * (m -> pos)[2] * (m -> m);
        kinetic += 0.5 * (m -> m) * pow((m -> vel).norm(), 2);
    }
        
    for (titan::Spring * s : sim.springs) {
        potential_s += s -> _k * pow((s -> _left -> pos - s -> _right -> pos).norm() - (s -> _rest), 2) / 2;
    }

    // std::cout << sim.time() << "," << potential_s << "," << kinetic << "," << potential_g << std::endl;
    // std::cout << "time " << sim.time() << ": gravitational potential energy is " << potential_g << " and spring potential is " << potential_s << " and kinetic energy is " << kinetic << " total is " << potential_g + potential_s + kinetic << std::endl;

    return potential_g + potential_s + kinetic;
}

struct energy_fixture : ::testing::Test {
    double tol = 0.999;
};

TEST_F(energy_fixture, energy_test) {
    titan::Simulation sim;
    sim.createPlane(titan::Vec(0, 0, 1), 0, 0, 0);
    sim.createLattice(titan::Vec(0, 0, 5), titan::Vec(4, 4, 4), 20, 20, 20);
    sim.setAllSpringConstantValues(100);
    sim.setAllDeltaTValues(0.0001);
    sim.setGlobalAcceleration(titan::Vec(0, 0, -9.8));
    sim.defaultRestLength();

    sim.start();

    double total_energy = energy(sim);
    while (sim.time() < 2) {
        sim.wait(0.25);
        EXPECT_NEAR(energy(sim), total_energy, total_energy * tol);
        sim.resume();
    }

    sim.stop();
}
