//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_MASS_H
#define LOCH_MASS_H

#include "vec.h"

class Mass;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
struct CUDA_MASS {
    CUDA_MASS() {};
    CUDA_MASS(Mass & mass);

    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2

#ifdef GRAPHICS
    Vec color;
#endif

    bool fixed; // is the mass position fixed?
    bool valid;
};

class Mass {
public:
    //Properties
    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    Vec pos; // position in m
    Vec vel; // velocity in m/s
    Vec acc; // acceleration in m/s^2
    Vec force; // force in kg m / s^2

    bool fixed; // is the mass position fixed?
    bool valid;

    int ref_count;
    struct CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

#ifdef GRAPHICS
    Vec color;
#endif

    //Set
    Mass();
    Mass(struct CUDA_MASS & mass);
    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);

    void setMass(double m) { this -> m = m; };
    void setPos(const Vec & pos) { this -> pos = pos; }
    void setVel(const Vec & vel) { this -> vel = vel; }
    void setAcc(const Vec & acc) { this -> acc = acc; }
    void setForce(const Vec & force) { this -> force = force; }
    void setDeltaT(double dt) { this -> dt = dt; }
    void translate(const Vec & displ) { this -> pos += displ; }
    void makeFixed() { fixed = true; }
    void makeMovable() { fixed = false; }

    //Get
    int isFixed() { return (fixed == true); }
    double getMass() { return m; }
    const Vec & getPosition() { return pos; }
    const Vec & getVelocity() { return vel; }
    const Vec & getAcceleration() { return acc; }
    const Vec & getForce() { return force; }
    double time() { return T; }
    double deltat() const { return dt; }
    void stepTime() { T += dt; }

    //Methods
    void update(); // update pos, vel, and acc based on force
    void addForce(const Vec &); // add force vector to current force
    void resetForce(); // set force = 0;

};

void decrementRefCount(Mass * m);

#endif //LOCH_MASS_H
