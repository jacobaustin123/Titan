#include "sim.h"

int main() {
    Simulation sim;

    Lattice * l1 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);

    sim.start(10.0); // 10 second runtime.

    while (sim.running()) {
        sim.stop(sim.time() + 1.0);
        sim.printPositions();

        if (sim.running()) {
            sim.resume();
        } else {
            break;
        }
    }
}