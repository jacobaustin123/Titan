#include "gtest/gtest.h"

#include <Titan/sim.h>

using titan::Vec;

struct simple_fixture : ::testing::Test {

};

TEST_F(simple_fixture, test_simple) {
    titan::Simulation sim;

    sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    sim.createBall(Vec(0, 0, 0), 2);

    sim.setBreakpoint(1.0);
    sim.start(); // 10 second runtime.
}
