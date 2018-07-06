//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_MASS_H
#define LOCH_MASS_H

#include "vec.h"
#include "object.h"

class Mass;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
struct CUDA_MASS {
    CUDA_MASS() = default;
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

    bool valid;

#ifdef CONSTRAINTS
    CUDA_LOCAL_CONSTRAINTS constraints;
#endif

};

#ifdef CONSTRAINTS
enum CONSTRAINT_TYPE {
    CONSTRAINT_PLANE, CONTACT_PLANE, BALL, DIRECTION
};
#endif

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

    bool valid;

    int ref_count;
    CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

#ifdef CONSTRAINTS
    LOCAL_CONSTRAINTS constraints;
    void addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num);
    void fix();
    void unfix();
#endif

#ifdef GRAPHICS
    Vec color;
#endif

    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);

private:
    //Set
    Mass();
    void operator=(CUDA_MASS & mass);

    friend class Simulation;
};

void decrementRefCount(Mass * m);

#endif //LOCH_MASS_H
