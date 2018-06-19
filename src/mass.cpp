//
// Created by Jacob Austin on 5/17/18.
//

#include "mass.h"

void Mass::update() { // update pos, vel, and acc based on force
    acc = force / m;
    vel = vel + acc * delta_t;
    pos = pos + vel * delta_t;
}

void Mass::addForce(const Vec & v) { // add force vector to current force
    force = force + v;
}

void Mass::resetForce() {
    force = Vec();
}