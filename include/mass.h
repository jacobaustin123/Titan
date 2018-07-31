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
    Mass();
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

#ifdef CONSTRAINTS
    LOCAL_CONSTRAINTS constraints;
    void addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num);
    void clearConstraints(CONSTRAINT_TYPE type);
    void clearAllConstraints();

    void setDrag(double C);
    void fix();
    void unfix();
#endif

#ifdef GRAPHICS
    Vec color;
#endif

    Mass(const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);

private:
    void decrementRefCount();

    CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

    void operator=(CUDA_MASS & mass);

    friend class Simulation;
    friend class Spring;
    friend struct CUDA_SPRING;
    friend class Container;
    friend class Lattice;
    friend class Cube;
};

class pyMass { // user mass class used to store real mass class properties in python and avoid ownership issues
public:
    //constructor
    pyMass() = default;
    pyMass(Mass * massp){ pointer = massp;}

    //Pointer to real C++ mass object
    Mass * pointer;

    //Properties
    double m() { return (pointer -> m);}  // mass in kg
    void m(int m) {pointer -> m = m;} // set mass
    double dt() { return (pointer -> dt);} // update interval
    double T() { return (pointer -> T);} // local time
    Vec pos() { return (pointer -> pos);} // position in m
    Vec vel() { return (pointer -> vel);} // velocity in m/s
    Vec acc() { return (pointer -> acc);} // acceleration in m/s^2
    Vec force() { return (pointer -> force);} // force in kg m / s^2

    void setm(double mass) { pointer -> m = mass;}  // mass in kg

};
#endif //LOCH_MASS_H
