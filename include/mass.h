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

    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);
private:
    //Set
    Mass();
    Mass(struct CUDA_MASS & mass);

    friend class Simulation;
};

void decrementRefCount(Mass * m);

#endif //LOCH_MASS_H
