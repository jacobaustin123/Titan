#include "sim.h"

int main()
{
    static Simulation sim;

    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(4, 4, 2), 3, 3, 2);

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

    sim.createBall(Vec(0, 0, 0), 2);

    sim.setBreakpoint(10.0);
    sim.run();

    return 0;
}
