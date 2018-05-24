//
// Created by Jacob Austin on 5/17/18.
//

#include "sim.h"

Simulation::~Simulation() {
    for (Mass * m : masses)
        delete m;

    for (Spring * s : springs)
        delete s;

    for (Constraint * c : constraints)
        delete c;

    for (ContainerObject * o : objs)
        delete o;
}

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

#include <cmath>

void Simulation::computeForces() {
    Spring * s = spring_arr;

    for (int i = 0; i < springs.size(); i++) { // update the forces
        s -> setForce();
        s++;
    }

//    Mass * m = mass_arr; // constraints and gravity
//    for (int i = 0; i < masses.size(); i++) {
//        for (Constraint * c : constraints) {
//            m -> addForce( c -> getForce(m -> getPosition()) ); // add force based on position relative to constraint
//        }
//
//        m -> addForce(Vec(0, 0, - m -> getMass() * G)); // add gravity
//
//        m++;
//    }
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

    delete [] spring_arr;
    delete [] mass_arr;
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
    RUNNING = 1;
    toArray();

    while (1) {
        T += dt;

        if (!bpts.empty() && *bpts.begin() <= T) {
            bpts.erase(bpts.begin());
            fromArray();
            RUNNING = 0;
            break;
        }

        computeForces(); // compute forces on all masses

//        printForces();

        Mass * m = mass_arr;
        for (int i = 0; i < masses.size(); i++) {
            if (m -> time() <= T) { // !m -> isFixed()
                m -> stepTime();
                m -> update();
            }

            m -> resetForce();

            m++;
        }
    }
}

int compareMass(const Mass * x, const Mass * y) { // Compare two masses' dts
    return x -> deltat() < y -> deltat() ? 0 : 1;
}

void Simulation::run() { // state initial simulation variables and initiate simulation for the first time
    T = 0; //Clock CPU and global simulation time to 0
    dt = (*std::min_element(masses.begin(), masses.end(), compareMass)) -> deltat(); //state simulation dt as the
            //minimum dt in the group of masses

    resume(); //Start the simulation for the first time
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

    objs.push_back(cube);

    return cube;
}

void Simulation::printPositions() {
    if (RUNNING) {
        Mass * m = mass_arr;
        for (int i = 0; i < masses.size(); i++) {
            std::cout << m -> getPosition() << std::endl;
            m++;
        }
    } else {
        for (Mass * m : masses) {
            std::cout << m->getPosition() << std::endl;
        }
    }

    std::cout << std::endl;
}

void Simulation::printForces() {
    if (RUNNING) {
        Mass * m = mass_arr;
        for (int i = 0; i < masses.size(); i++) {
            std::cout << m -> getForce() << std::endl;
            m++;
        }
    } else {
        for (Mass * m : masses) {
            std::cout << m->getForce() << std::endl;
        }
    }

    std::cout << std::endl;
}