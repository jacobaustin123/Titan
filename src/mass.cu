//
// Created by Jacob Austin on 5/17/18.
//

#include "mass.h"

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