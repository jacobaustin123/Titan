#include <Titan/sim.h> 
#include <iostream>
#include <vector>
#include <map>
#include <tuple>

#include "gtest/gtest.h"

#define SIZE 5.0
#define SPACE 3.0
#define NUM_X 10
#define NUM_Y 10
#define DENSITY 5

using titan::Vec;

struct multiagent_fixture : ::testing::Test {

};

TEST_F(multiagent_fixture, multiagent_test) {
    titan::Simulation sim;
    // sim.setViewport(Vec(0, 0, 10 * SIZE), Vec(NUM_X * (SIZE + SPACE) / 2, NUM_Y * (SIZE + SPACE) / 2, SIZE / 2), Vec(0, 0, 1)); 
    
    sim.setGlobalAcceleration(Vec(0, 0, -9.8));

    titan::Lattice * grid[NUM_Y][NUM_X];

    for (int i = 0; i < NUM_Y; i++) {
        for (int j = 0; j < NUM_X; j++) {
            grid[i][j] = sim.createLattice(Vec((SIZE + SPACE) * j + SIZE / 2, (SIZE + SPACE) * i + SIZE / 2, SIZE / 2), Vec(SIZE, SIZE, SIZE), DENSITY, DENSITY, DENSITY);
        }
    }
    titan::Mass * m1, *m2;
    titan::Spring * s1;

    std::vector<std::tuple<int, int>> right = {std::make_tuple(100, 0), std::make_tuple(120, 20), std::make_tuple(104, 4), std::make_tuple(124, 24)};
    std::vector<std::tuple<int, int>> up = {std::make_tuple(20, 0), std::make_tuple(120, 100), std::make_tuple(124, 104), std::make_tuple(24, 4)};

    for (int i = 0; i < NUM_Y - 1; i++) {
        for (int j = 0; j < NUM_X - 1; j++) {
            for (auto pair : right) {
                m1 = grid[i][j] -> masses[std::get<0>(pair)];
                m2 = grid[i][j+1] -> masses[std::get<1>(pair)];

                // std::cout << m1 -> pos << " " << m2 -> pos << std::endl;

                s1 = sim.createSpring(m1, m2);
                s1 -> _k = 0.01;
                s1 -> defaultLength();
            }

            for (auto pair : up) {
                m1 = grid[i][j] -> masses[std::get<0>(pair)];
                m2 = grid[i+1][j] -> masses[std::get<1>(pair)];
                s1 = sim.createSpring(m1, m2);
                s1 -> _k = 0.01;
                s1 -> defaultLength();
                //s1 -> _rest = s1 -> _rest + 0.0001;
            }
        }
    }

    sim.defaultRestLengths();
    sim.start();
    sim.pause(1.0);

    sim.getAll();
    sim.stop();
    // sim.printPositions();
}
