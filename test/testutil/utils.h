#pragma once

#include <Titan/sim.h>
#include <iostream>

using namespace std;

namespace titan {

namespace test {

inline double energy(titan::Simulation & sim) {
    double potential_g = 0;
    double potential_s = 0;
    double kinetic = 0;
    sim.getAll();

    for (titan::Mass * m : sim.masses) {
        potential_g += 9.8 * (m -> pos)[2] * (m -> m);
        kinetic += 0.5 * (m -> m) * pow((m -> vel).norm(), 2);
    }
        
    for (titan::Spring * s : sim.springs) {
        potential_s += s -> _k * pow((s -> _left -> pos - s -> _right -> pos).norm() - (s -> _rest), 2) / 2;
    }

    // cout << sim.time() << "," << potential_s << "," << kinetic << "," << potential_g << endl;

    return potential_s + kinetic + potential_g;
}

inline titan::Vec momentum(titan::Simulation & sim) {
    titan::Vec linear_momentum = titan::Vec(0, 0, 0);
    titan::Vec angular_momentum = titan::Vec(0, 0, 0);
    sim.getAll();

    for (titan::Mass * m : sim.masses) {
        linear_momentum += m -> m * m -> vel;
        angular_momentum += titan::cross(m -> m * m -> vel, m -> pos);
    }

    return linear_momentum + angular_momentum;
}

} // namespace test

} // namespace titan