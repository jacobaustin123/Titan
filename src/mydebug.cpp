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
    m3 -> setPos(Vec(-1, -1, -1));
    m.push_back(m3);

    Mass * m4 = sim.createMass();
    m4 -> setPos(Vec(1, -1, -1));
    m.push_back(m4);

    Mass * m5 = sim.createMass();
    m5 -> setPos(Vec(1, 1, 1));
    m.push_back(m5);

    Mass * m6 = sim.createMass();
    m6 -> setPos(Vec(-1, 1, 1));
    m.push_back(m6);

    Mass * m7 = sim.createMass();
    m7 -> setPos(Vec(-1, -1, 1));
    m.push_back(m7);

    Mass * m8 = sim.createMass();
    m8 -> setPos(Vec(1, -1, 1));
    m.push_back(m8);

    Spring * s1 = sim.createSpring(m1, m2, 1.0, 2.0); // bottom
    Spring * s2 = sim.createSpring(m2, m3, 1.0, 2.0);
    Spring * s3 = sim.createSpring(m3, m4, 1.0, 2.0);
    Spring * s4 = sim.createSpring(m4, m1, 1.0, 2.0);

    Spring * s5 = sim.createSpring(m1, m5, 1.0, 2.0); // up down
    Spring * s6 = sim.createSpring(m2, m6, 1.0, 2.0);
    Spring * s7 = sim.createSpring(m3, m7, 1.0, 2.0);
    Spring * s8 = sim.createSpring(m4, m8, 1.0, 2.0);

    Spring * s9 = sim.createSpring(m5, m6, 1.0, 2.0); // top
    Spring * s10 = sim.createSpring(m6, m7);
    Spring * s11 = sim.createSpring(m7, m8);
    Spring * s12 = sim.createSpring(m8, m5);

    Spring * s13 = sim.createSpring(m1, m7);
    Spring * s14 = sim.createSpring(m3, m5);

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
    sim.run(); // could fork, child process would run kernels and then die when it reaches a breakpoint

    while (sim.time() < 100.0) {
        sim.setBreakpoint(sim.time() + 1);
        sim.resume();
        sim.printPositions();
    }
}

