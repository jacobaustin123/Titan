//
// Created by Jacob Austin on 5/17/18.
//

#include "mass.h"

Mass::Mass() {
    m = 1.0;
    fixed = false;
    dt = 0.0001;
    T = 0;
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
} // constructor TODO fix timing

Mass::Mass(struct CUDA_MASS & mass) {
    m = mass.m;
    dt = mass.dt;
    T = mass.T;
    pos = mass.pos;
    vel = mass.vel;
    acc = mass.acc;
    force = mass.force;
    fixed = mass.fixed;
    valid = mass.valid;

    ref_count = 0;
    arrayptr = nullptr;

#ifdef GRAPHICS
    color = mass.color;
#endif
}

Mass::Mass(const Vec & position, double mass, bool fixed, double dt) {
    m = mass;
    pos = position;

    this -> fixed = fixed;
    this -> dt = dt;

    T = 0;
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
}

void Mass::update() { // update pos, vel, and acc based on force
    acc = force / m;
    vel = vel + acc * dt;
    pos = pos + vel * dt;
}

void Mass::addForce(const Vec & v) { // add force vector to current force
    force = force + v;
}

void Mass::resetForce() {
    force = Vec();
}

CUDA_MASS::CUDA_MASS(Mass &mass) {
    m = mass.m;
    dt = mass.dt;
    T = mass.T;
    pos = mass.pos;
    vel = mass.vel;
    acc = mass.acc;
    force = mass.force;
    fixed = mass.fixed;
    valid = true;

#ifdef GRAPHICS
    color = mass.color;
#endif
}