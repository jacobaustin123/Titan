#include "vec.h"
#include "sim.h"
#include <iostream>

int main() {
    Simulation sim; // initialize scene object

    std::vector<Mass *> m;

    Mass * m1 = sim.createMass();
    m1 -> setPos(Vec(1, 1, -1));
    m.push_back(m1);

    Mass * m2 = sim.createMass();
    m2 -> setPos(Vec(-1, 1, -1));
    m.push_back(m2);

    Mass * m3 = sim.createMass();
    m3 -> setPos(Vec(1, -1, -1));
    m.push_back(m3);

    Mass * m4 = sim.createMass();
    m4 -> setPos(Vec(-1, -1, -1));
    m.push_back(m4);

    Mass * m5 = sim.createMass();
    m5 -> setPos(Vec(1, 1, 1));
    m.push_back(m5);

    Mass * m6 = sim.createMass();
    m6 -> setPos(Vec(-1, 1, 1));
    m.push_back(m6);

    Mass * m7 = sim.createMass();
    m7 -> setPos(Vec(1, -1, 1));
    m.push_back(m7);

    Mass * m8 = sim.createMass();
    m8 -> setPos(Vec(-1, -1, 1));
    m.push_back(m8);

    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            Spring * s = sim.createSpring();
            s -> setK(1000.0);
            s -> setRestLength(2);
            s -> setMasses(m[i], m[j]);
        }
    }

    sim.printPositions();

    sim.setBreakpoint(1);
    sim.run();

    while (sim.time() < 100.0) {
        sim.setBreakpoint(sim.time() + 1);
        sim.resume();
        sim.printPositions();
    }



}