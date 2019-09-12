#include <Titan/sim.h> 
#include <iostream>
#include <vector>
#include <map>
#include <tuple>

#define SIZE 5.0
#define SPACE 3.0
#define NUM_X 10
#define NUM_Y 10
#define DENSITY 5

int main() {
    Simulation sim;

    // Lattice * l2 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    // sim.setAllSpringConstantValues(1E5);
    sim.setViewport(Vec(0, 0, 5 * SIZE), Vec(NUM_X * (SIZE + SPACE) / 2, NUM_Y * (SIZE + SPACE) / 2, SIZE / 2), Vec(0, 0, 1)); 
    sim.createPlane(Vec(0, 0, 1), 0);

    sim.setGlobalAcceleration(Vec(0, 0, -9.8));

    Lattice * grid[NUM_Y][NUM_X];

    for (int i = 0; i < NUM_Y; i++) {
        for (int j = 0; j < NUM_X; j++) {
            grid[i][j] = sim.createLattice(Vec((SIZE + SPACE) * j + SIZE / 2, (SIZE + SPACE) * i + SIZE / 2, SIZE / 2), Vec(SIZE, SIZE, SIZE), DENSITY, DENSITY, DENSITY);
        }
    }

    // int j = 0;
    // std::vector<Vec> values = {Vec(0, 0, 0), Vec(5, 0, 0), Vec(0, 5, 0), Vec(0, 0, 5), Vec(5, 5, 0), Vec(0, 5, 5), Vec(5, 0, 5), Vec(5, 5, 5)};
    // Mass * m;
    // for (int i = 0; i < grid[0][0] -> masses.size(); i++) {
    //     m = grid[0][0] -> masses[i];
    //     for (Vec v : values) {
    //         if (m -> pos == v) {
    //             std::cout << i << " " << v << std::endl;
    //         }
    //     }

    //     // if (m -> pos[0] == value[0] && m -> pos[1] == value[1] && m -> pos[2] == value[2]) {
    //     //     j = i;
    //     //     break;
    //     // }
    // }

    // std::cout << j << " " << value << std::endl;

    Mass * m1, *m2;
    Spring * s1, *s2;

    std::vector<std::tuple<int, int>> right = {std::make_tuple(100, 0), std::make_tuple(120, 20), std::make_tuple(104, 4), std::make_tuple(124, 24)};
    std::vector<std::tuple<int, int>> up = {std::make_tuple(20, 0), std::make_tuple(120, 100), std::make_tuple(124, 104), std::make_tuple(24, 4)};

    for (int i = 0; i < NUM_Y - 1; i++) {
        for (int j = 0; j < NUM_X - 1; j++) {
            for (auto pair : right) {
                m1 = grid[i][j] -> masses[std::get<0>(pair)];
                m2 = grid[i][j+1] -> masses[std::get<1>(pair)];

                // std::cout << m1 -> pos << " " << m2 -> pos << std::endl;

                s1 = sim.createSpring(m1, m2);
                s1 -> _k = 20;
                s1 -> defaultLength();
            }

            for (auto pair : up) {
                m1 = grid[i][j] -> masses[std::get<0>(pair)];
                m2 = grid[i+1][j] -> masses[std::get<1>(pair)];
                s1 = sim.createSpring(m1, m2);
                s1 -> _k = 20;
                s1 -> defaultLength();
            }
        }
    }

    sim.start();
}
