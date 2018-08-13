#include "../Loch/include/Loch/sim.h"

int main() {
    Simulation sim;

    Lattice * l2 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
//    Container * l2 = sim.importFromSTL("C:/Users/CreativeMachines/Desktop/Part3.STL", 10, 5); // 10 points per square meter, 5 rays per point.
//    sim.setViewport(Vec(5, -10, 3), Vec(0, 0, 2), Vec(0, 0, 1));

    sim.setAllSpringConstantValues(1E5);
    l2 -> rotate(Vec(0, 0, 1), -0.78);

    std::cout << sim.masses.size() << " " << sim.springs.size() << std::endl;
    sim.createPlane(Vec(0, 0, 1), 0);

    double runtime = 10.0;
    sim.setGlobalForce(Vec(0, 0, -0.1));

    sim.start(); // 10 second runtime.

    while (true) {
        sim.pause(sim.time() + 1);
        sim.get(l2);
        l2 -> rotate(Vec(0, 0, 1), 0.5);
        sim.set(l2);
        sim.resume();
    }
}