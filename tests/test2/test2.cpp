#include <Loch/sim.h>

int main()
{
    static Simulation sim;

//    Beam * l1 = sim.createBeam(Vec(0, 0, 5), Vec(10, 4, 2), 10, 5, 2);
    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(10, 4, 2), 10, 5, 2);

    sim.setSpringConstant(10000);
    sim.setMassDeltaT(0.0001);

//    sim.createPlane(Vec(0, 0, 1), 0);
    sim.createBall(Vec(0, 0, 0), 2);

    sim.setBreakpoint(10.0);
    sim.run();

    return 0;
}
