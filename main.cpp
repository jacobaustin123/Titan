#include "../Loch/include/Loch/sim.h"

int main() {
    Simulation sim;

//    Lattice * l1 = sim.createLattice(Vec(0, 0, 10), Vec(5, 5, 5), 10, 10, 10);
    Container * l2 = sim.importFromSTL("C:/Users/CreativeMachines/Desktop/Part2.STL", 80, 5); // 10 points per square meter, 5 rays per point.
    l2 -> translate(-Vec(0, 0, 3));

    sim.setSpringConstant(1E5);

    std::cout << sim.masses.size() << " " << sim.springs.size() << std::endl;
    sim.createPlane(Vec(0, 0, 1), 0);

    double runtime = 10.0;

    sim.setBreakpoint(runtime);
    sim.start(); // 10 second runtime.
}