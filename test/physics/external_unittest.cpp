#include <Titan/sim.h> 
#include <chrono>
#include <iostream>

#include <testutil/utils.h>

#include "gtest/gtest.h"

using namespace std;

struct external_fixture : ::testing::Test {
    double tol = 0.000001;
    double z_tol = 0.1;
};

TEST_F(external_fixture, external_test) {
    titan::Simulation sim;
    titan::Mass * m1 = sim.createMass(titan::Vec(1, 0, 1));
    sim.setTimeStep(0.0001);
    m1 -> setExternalForce(m1 -> m * titan::Vec(0, 0, -9.8));
    sim.setGlobalAcceleration(titan::Vec(0, 0, 0.0));

    sim.start();
    while (sim.time() < 5) {
        sim.wait(0.1);
        sim.getAll();
        EXPECT_NEAR(m1 -> pos[0], 1, tol);
        EXPECT_NEAR(m1 -> pos[1], 0, tol);
        EXPECT_NEAR(m1 -> pos[2], 1 + 0.5 * -9.8 * pow(sim.time(), 2), z_tol);

        sim.resume();
    }

    sim.stop();
}

TEST_F(external_fixture, external_acc_test) {
    titan::Simulation sim;
    titan::Mass * m1 = sim.createMass(titan::Vec(1, 0, 1));
    sim.setTimeStep(0.0001);
    m1 -> setExternalForce(m1 -> m * titan::Vec(0, 0, 0));
    sim.setGlobalAcceleration(titan::Vec(0, 0, -9.8));

    sim.start();
    while (sim.time() < 5) {
        sim.wait(0.1);
        sim.getAll();
        EXPECT_NEAR(m1 -> pos[0], 1, tol);
        EXPECT_NEAR(m1 -> pos[1], 0, tol);
        EXPECT_NEAR(m1 -> pos[2], 1 + 0.5 * -9.8 * pow(sim.time(), 2), z_tol);

        sim.resume();
    }

    sim.stop();
}

TEST_F(external_fixture, external_acc_test2) {
    titan::Simulation sim;
    titan::Mass * m1 = sim.createMass(titan::Vec(1, 0, 1));
    sim.setTimeStep(0.0001);
    sim.setGlobalAcceleration(titan::Vec(0, 0, -9.8));

    sim.start();
    while (sim.time() < 5) {
        sim.wait(0.1);
        sim.getAll();
        EXPECT_NEAR(m1 -> pos[0], 1, tol);
        EXPECT_NEAR(m1 -> pos[1], 0, tol);
        EXPECT_NEAR(m1 -> pos[2], 1 + 0.5 * -9.8 * pow(sim.time(), 2), z_tol);

        sim.resume();
    }

    sim.stop();
}