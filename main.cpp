#include "include/Loch/sim.h"

int main() {
    Simulation sim;
    sim.createLattice(Vec(0, 0, 5), Vec(5, 5, 5), 5, 5, 5);
    sim.createPlane(Vec(0, 0, 1), 0);
    sim.start();
}