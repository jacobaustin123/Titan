//
//  cube.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include <stdio.h>
#include <iostream>
#include <vector>
#include "cube.hpp"

void Mass::update() { // update P, V, and A based on F
    A = F / m;
    V = V + A * dt;
    P = P + V * dt;
}

void Mass::addForce(Vec v) { // add force vector to current force
    F = F + v;
}

void Mass::resetForce() {
    F = Vec();
}

Vec Spring::getForce() { // computes force on right object. left force is - right force.
    Vec temp = (right -> P) - (left -> P);
    return temp * k * (L0 - temp.norm()) / temp.norm();
}

Cube::Cube() { // create masses at the four corners
    mass = 0.8;
    Vec temp;
    temp = Vec(0.05, 0.05, 0.2);
    masses[0] = Mass(0.1, temp);
    temp = Vec(0.05, -0.05, 0.2);
    masses[1] = Mass(0.1, temp);
    temp = Vec(-0.05, -0.05, 0.2);
    masses[2] = Mass(0.1, temp);
    temp = Vec(-0.05, 0.05, 0.2);
    masses[3] = Mass(0.1, temp);
    temp = Vec(0.05, 0.05, 0.1);
    masses[4] = Mass(0.1, temp);
    temp = Vec(0.05, -0.05, 0.1);
    masses[5] = Mass(0.1, temp);
    temp = Vec(-0.05, -0.05, 0.1);
    masses[6] = Mass(0.1, temp);
    temp = Vec(-0.05, 0.05, 0.1);
    masses[7] = Mass(0.1, temp);
    
    
    for (int i = 0; i < 8; i++) { // add the appropriate springs
        for (int j = i + 1; j < 8; j++) {
            springs.push_back(Spring(k, (masses[i].P - masses[j].P).norm(), &masses[i], &masses[j]));
        }
    }
}

void Scene::next() {
    Vec force;
    for (Spring s : cube.springs) { // update the forces
        force = s.getForce();
        s.right -> addForce(force);
        s.left -> addForce(-force);
    }
    
    for (int j = 0; j < 8; j++) {
        force = Vec(0, 0, -cube.masses[j].m * G); // add gravity
        cube.masses[j].addForce(force);
        if (cube.masses[j].P[2] < 0) { // check for constraints
            force = Vec(0, 0, - DISPL_CONST * cube.masses[j].P[2]);
            cube.masses[j].addForce(force);
        }
        cube.masses[j].update(); // update positions, velocities, and accelerations accordingly
        cube.masses[j].resetForce(); // reset forces on each mass
    }
}

void Scene::simulate() { // repeatedly run next
    for (int i = 0; i < frames; i++) {
        next();
    }
}
