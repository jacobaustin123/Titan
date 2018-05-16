//
//  cube.hpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

// in scene, initialize Cube with 8 masses, 24 springs
// open OpenGL window, initialize scene
// simulate method which will loop over all springs, calculate force on each mass
// compute gravity on each mass, check for negative z values
// update position, velocity, and acceleration of each mass
// draw new scene with new positions
// repeat until all velocity is below some epsilon

#ifndef cube_hpp
#define cube_hpp

#include <stdio.h>
#include <vector>
#include "vec.hpp"

static float G = 9.81;
static float dt = 0.001;
static float DISPL_CONST = 100000;
static float k = 10000;

class Mass {
public:
    float m; // mass in kg
    Vec P; // position in m
    Vec V; // velocity in m/s
    Vec A; // acceleration in m/s^2
    Vec F; // force in kg m / s^2
    
    Mass() {};
    Mass(float i):m(i) {}; // defaults everything to 0
    Mass(float i, Vec & p):m(i), P(p) {};
    Mass(float i, Vec & p, Vec & v, Vec & a, Vec & f): m(i), P(p), V(v), A(a), F(f) {}; // constructor
    
    void update(); // update P, V, and A based on F
    void addForce(Vec); // add force vector to current force
    void resetForce(); // set F = 0;
};

class Spring {
public:
    float k; // spring constant (N/m)
    float L0; // spring rest length (meters)
    Mass * left; // pointer to left mass object
    Mass * right; // pointer to right mass object
    
    Spring(float _k, float _L0, Mass * l, Mass * r):k(_k), L0(_L0), left(l), right(r) {};
    Vec getForce(); // computes force on right object. left force is - right force.
};


// should have Object base class, with vector<Mass> m_vec, which we push masses onto.
// should have vector<Spring> s_vec, which contians springs associated with the object.
// should have addSprings method, which iterates over the list of masses and connects
// them with springs, which are added to the list.

// can have Cube child class, which takes an origin Vec parameter and a size parameter, and creates
// an object class with those properties.

// Object base class should have a render method (and a buffer variable), which updates the buffer
// and draws the points/lines. Also would have an update method which updates all of its masses.

// then then in the main method, we just initialize the window, and look over the masses, updating
// and rendering each one.


// for CUDA, we want to be able to split large lists of operations into components which are multiprocessed
// by the GPU. for this reason, we might want to store all springs in one list, and iterate over the full list
// to update everything.

class Cube {
public:
    float mass; // cube mass
    Mass masses[8]; // list of masses at corners
    std::vector<Spring> springs; // list of springs connecting objects
    
    Cube();
};

class Scene {
public:
    int frames;
    Cube cube;
    Scene(int frames = 20) : frames(frames) {} ;
    void simulate();
    void next();
};

#endif /* cube_hpp */

