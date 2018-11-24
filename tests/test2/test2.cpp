#include <Titan/sim.h>

int main() {
    Simulation sim;

    Lattice * l1 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    sim.createBall(Vec(0, 0, 0), 2);

    sim.start(); // 10 second runtime.
}
