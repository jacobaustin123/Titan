#include <Titan/sim.h> 
#include <iostream>

void energy(Simulation & sim) {
    double potential_g = 0;
    double potential_s = 0;
    double kinetic = 0;
    sim.getAll();

    for (Mass * m : sim.masses) {
        potential_g += 9.8 * (m -> pos)[2] * (m -> m);
        kinetic += 0.5 * (m -> m) * pow((m -> vel).norm(), 2);
    }
        
    for (Spring * s : sim.springs) {
        potential_s += s -> _k * pow((s -> _left -> pos - s -> _right -> pos).norm() - (s -> _rest), 2) / 2;
    }

    std::cout << "time " << sim.time() << ": gravitational potential energy is " << potential_g << " and spring potential is " << potential_s << " and kinetic energy is " << kinetic << " total is " << potential_g + potential_s + kinetic << std::endl;
}

int main() {
    Simulation sim;
    sim.createPlane(Vec(0, 0, 1), 0, 0, 0);
    Lattice * l1 = sim.createLattice(Vec(0, 0, 5), Vec(4, 4, 4), 10, 10, 10);
    //Mass * m1 = sim.createMass(Vec(0, 0, 1));
    sim.setAllSpringConstantValues(100);
    sim.setAllDeltaTValues(0.0001);
    sim.setGlobalAcceleration(Vec(0, 0, -9.8));
    sim.defaultRestLength();

    sim.start();

    while (sim.time() < 10) {
        sim.wait(0.5);
        energy(sim);
        sim.resume();
    }

    sim.stop();

    return 0;
}
