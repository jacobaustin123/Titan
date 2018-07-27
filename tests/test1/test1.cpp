#include "sim.h"

int main()
{
    Simulation sim;

    Beam * l1 = sim.createBeam(Vec(0, 0, 5), Vec(4, 4, 2), 5, 5, 3);
    Mass * m1 = sim.createMass(Vec(0, 0, 10));
    Mass * m2 = sim.createMass(Vec(0, 0, 11));

    Spring * s1 = sim.createSpring(m1, m2);

    sim.setSpringConstant(1000);
    sim.setMassDeltaT(0.0001);

//    sim.createBall(Vec(0, 0, 0), 2);
    sim.createPlane(Vec(0, 0, 1), 0); // ax + by + cz = d

    sim.setBreakpoint(10.0);
    sim.run();

    return 0;
}
