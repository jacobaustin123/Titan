#include "sim.h"

int main() {
    static Simulation sim;

    Lattice * l1 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    sim.createBall(Vec(0, 0, 0), 2);

    l1 -> masses[0] -> addConstraint(CONSTRAINT_PLANE, Vec(0, 0, 1), 0);

    double runtime = 10.0;

    sim.start(runtime); // 10 second runtime.

    while (sim.running()) {
        sim.pause(sim.time() + 1.0);

        if (sim.time() > runtime) {
            sim.stop();
            break;
        } else {
            sim.resume();
        }
    }
}