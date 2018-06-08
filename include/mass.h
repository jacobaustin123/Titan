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

    int fixed; // is the mass position fixed?
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
    int fixed; // is the mass position fixed?
    struct CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

#ifdef GRAPHICS
    Vec color;

    //Set
    Mass() { m = 1.0; fixed = 0; dt = 0.01; T = 0; color = Vec(1.0, 0.2, 0.2); } // constructor
    Mass(struct CUDA_MASS & mass)
            { m = mass.m; dt = mass.dt; T = mass.T; pos = mass.pos; vel = mass.vel; acc = mass.acc; force = mass.force; fixed = mass.fixed; color = mass.color; }
    Mass(const Vec & position, double mass = 1.0, int fixed = 0, double dt = 0.01) :
            m(mass), pos(position), fixed(fixed), dt(dt), T(0), color(Vec(1.0, 0.2, 0.2)) {}; // defaults everything
#else
    Mass() { m = 1.0; fixed = 0; dt = 0.01; T = 0; } // constructor
    Mass(struct CUDA_MASS & mass)
    { m = mass.m; dt = mass.dt; T = mass.T; pos = mass.pos; vel = mass.vel; acc = mass.acc; force = mass.force; fixed = mass.fixed;}
    Mass(const Vec & position, double mass = 1.0, int fixed = 0, double dt = 0.01) :
            m(mass), pos(position), fixed(fixed), dt(dt), T(0) {}; // defaults everything
#endif

    void setMass(double m) { this -> m = m; };
    void setPos(const Vec & pos) { this -> pos = pos; }
    void setVel(const Vec & vel) { this -> vel = vel; }
    void setAcc(const Vec & acc) { this -> acc = acc; }
    void setForce(const Vec & force) { this -> force = force; }
    void setDeltaT(double dt) { this -> dt = dt; }
    void translate(const Vec & displ) { this -> pos += displ; }
    void makeFixed() { fixed = 1; }
    void makeMovable() { fixed = 0; }

    //Get
    int isFixed() { return (fixed == 1); }
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


#endif //LOCH_MASS_H
