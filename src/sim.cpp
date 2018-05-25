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

Spring * Simulation::createSpring(Mass * m1, Mass * m2, double k, double len) {
    Spring * s = new Spring(m1, m2, k, len);
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

    Mass * m = mass_arr; // constraints and gravity
    for (int i = 0; i < masses.size(); i++) {
        for (Constraint * c : constraints) {
            m -> addForce( c -> getForce(m -> getPosition()) ); // add force based on position relative to constraint
        }

        m -> addForce(Vec(0, 0, - m -> getMass() * G)); // add gravity

        m++;
    }
}

CUDA_MASS * Simulation::massToArray() {
    CUDA_MASS * d_mass;
    cudaMalloc(d_mass, sizeof(CUDA_MASS) * masses.size());

    CUDA_MASS * iter = d_mass;

    for (Mass * m : masses) {
        CUDA_MASS mass(*m);
        cudaMemcpy(iter, mass, sizeof(CUDA_MASS));
        m -> arrayptr = iter;
        iter++;
    }

    this -> mass_arr = d_mass;

    return d_mass;
}

void Simulation::toArray() {
    CUDA_MASS * d_mass = massToArray();
    CUDA_SPRING * d_spring;
    cudaMalloc(d_spring, sizeof(CUDA_SPRING) * springs.size());

    CUDA_SPRING * spring_iter = d_spring;

    for (Spring * s : springs) {
        CUDA_SPRING spr(*s);
        cudaMemcpy(spring_iter, spr, sizeof(CUDA_SPRING));
        cudaMemcpy(spring_iter, s -> _left -> arrayptr, sizeof(CUDA_MASS *));
        cudaMemcpy((char *) spring_iter + sizeof(CUDA_MASS *), s -> _right -> arrayptr, sizeof(CUDA_MASS *));
        spring_iter++;
    }

    this -> spring_arr = d_spring;
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

void Simulation::run() { // repeatedly run next
    T = 0;
    dt = 0.01; // (*std::min_element(masses.begin(), masses.end(), compareMass)) -> deltat();
    
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
    for (Mass * m : masses) {
            std::cout << m->getPosition() << std::endl;
    }

    std::cout << std::endl;
}

void Simulation::printForces() {
    for (Mass * m : masses) {
        std::cout << m->getForce() << std::endl;
    }

    std::cout << std::endl;
}
