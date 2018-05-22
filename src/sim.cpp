//
// Created by Jacob Austin on 5/17/18.
//

#include "sim.h"

Mass * Simulation::createMass() {
    Mass * m = new Mass();
    masses.push_back(m);
    return m;
}

Spring * Simulation::createSpring() {
    Spring * s = new Spring();
    springs.push_back(s);
    return s;
}

void Simulation::setBreakpoint(double time) {
    bpts.insert(time);
}

void Simulation::computeForces() {
    for (int i = 0; i < springs.size(); i++) { // update the forces
        Spring * s = spring_arr + i;
        s -> setForce();
//        std::cout << s._left->getForce() << std::endl;
    }

    for (int i = 0; i < springs.size(); i++) {
        Mass * m = mass_arr + i;
        for (Constraint * c : constraints) {
//            std::cout << c -> getForce(m -> getPosition()) << std::endl;
            m -> addForce( c -> getForce(m -> getPosition()) ); // add force based on position relative to constraint
        }

        m -> addForce(Vec(0, 0, - m -> getMass() * G)); // add gravity
    }
}

Mass * Simulation::massToArray() {
    Mass * data = new Mass[masses.size()];
    Mass * iter = data;

    for (Mass * m : masses) {
        memcpy(iter, m, sizeof(Mass));
        m -> arrayptr = iter;
//        std::cout << iter -> getPosition() << m -> getPosition() << std::endl;
        iter++;
    }

    this->mass_arr = data;

    return data;
}

void Simulation::toArray() {
    Mass * mass_data = massToArray();
    Spring * spring_data = new Spring[springs.size()];

    Spring * spring_iter = spring_data;

    for (Spring * s : springs) {
        memcpy(spring_iter, s, sizeof(Spring));
        spring_iter -> setMasses(s -> _left -> arrayptr, s -> _right -> arrayptr);
        spring_iter++;
    }

    this -> spring_arr = spring_data;
}

void Simulation::fromArray() {
    massFromArray();

//    Spring * data = spring_arr;
//
//    for (Spring * s : springs) {
//        memcpy(s, data, sizeof(Spring));
//        s -> setMasses(masses[(data -> _left) - mass_arr], masses[(data -> _right) - mass_arr]);
//        data += sizeof(Spring);
//    }
}

void Simulation::massFromArray() {
    Mass * data = mass_arr;

    for (Mass * m : masses) {
        memcpy(m, data, sizeof(Mass));
        data ++;
    }
}

Spring * Simulation::springToArray() {
    Spring * data = new Spring[springs.size()];
    Spring * iter = data;

    for (Spring * s : springs) {
        memcpy(iter, s, sizeof(Spring));
        iter++;
    }

    this -> spring_arr = data;

    return data;
}

void Simulation::springFromArray() {
    Spring * data = spring_arr;

    for (Spring * s : springs) {
        memcpy(s, data, sizeof(Spring));
        data++;
    }
}

void Simulation::resume() {
    toArray();

    while (1) {
        T += dt;

        if (!bpts.empty() && *bpts.begin() <= T) {
            bpts.erase(bpts.begin());
            fromArray();
            break;
        }

        computeForces(); // compute forces on all masses
        for (int i = 0; i < masses.size(); i++) {

            Mass * m = mass_arr + i;
            if (!m -> isFixed() && m -> time() <= T) {
                m -> stepTime();
                m -> update();
            }

            m -> resetForce();
        }
    }
}

int compareMass(const Mass & x, const Mass & y) {
    return x.deltat() < y.deltat() ? 0 : 1;
}

void Simulation::run() { // repeatedly run next
    T = 0;
    dt = 0.01; //std::min_element(mass_arr, mass_arr + masses.size(), compareMass) -> deltat();

    resume();
}

Plane * Simulation::createPlane(const Vec & abc, double d ) { // creates half-space ax + by + cz < d
    Plane * new_plane = new Plane(abc, d);
    constraints.push_back(new_plane);
    return new_plane;
}

Cube * Simulation::createCube(const Vec & center, double side_length) { // creates half-space ax + by + cz < d
    Cube * cube = new Cube(center, side_length);
    for (Mass * m : cube -> masses) {
        masses.push_back(m);
    }

    for (Spring * s : cube -> springs) {
        springs.push_back(s);
    }

    return cube;
}