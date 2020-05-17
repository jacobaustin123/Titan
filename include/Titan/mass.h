//
// Created by Jacob Austin on 5/17/18.
//

#ifndef TITAN_MASS_H
#define TITAN_MASS_H

#include "vec.h"
#include "object.h"

namespace titan {

class Mass;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory
struct CUDA_MASS {
    CUDA_MASS() = default;
    CUDA_MASS(Mass & mass);

    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    titan::Vec pos; // position in m
    titan::Vec vel; // velocity in m/s
    titan::Vec acc; // acceleration in m/s^2
    titan::Vec force; // force in kg m / s^2

#ifdef RK2
    titan::Vec __rk2_backup_pos;
    titan::Vec __rk2_backup_vel;
#endif

#ifdef GRAPHICS
    titan::Vec color;
#endif

    bool valid;

#ifdef CONSTRAINTS
    CUDA_LOCAL_CONSTRAINTS constraints;
#endif

};

class Mass {
public:
    //Properties
    double m; // mass in kg
    double dt; // update interval
    double T; // local time
    titan::Vec pos; // position in m
    titan::Vec vel; // velocity in m/s
    titan::Vec acc; // acceleration in m/s^2
    titan::Vec force; // force in kg m / s^2

#ifdef RK2
    titan::Vec __rk2_backup_pos;
    titan::Vec __rk2_backup_vel;
#endif

    Mass(const titan::Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001);
#ifdef CONSTRAINTS
    void addConstraint(CONSTRAINT_TYPE type, const titan::Vec & vec, double num);
    void clearConstraints(CONSTRAINT_TYPE type);
    void clearConstraints();

    void setDrag(double C);
    void fix();
    void unfix();
#endif
    
#ifdef GRAPHICS
    titan::Vec color;
#endif

private:
    bool valid;
    int ref_count;

    void decrementRefCount();

    CUDA_MASS * arrayptr; //Pointer to struct version for GPU cudaMemAlloc

    Mass();
    void operator=(CUDA_MASS & mass);

    friend class Simulation;
    friend class Spring;
    friend struct CUDA_SPRING;
    friend struct CUDA_MASS;
    friend class Container;
    friend class Lattice;
    friend class Beam;
    friend class Cube;

#ifdef CONSTRAINTS
    LOCAL_CONSTRAINTS constraints;

#endif

};

} // namespace titan

#endif //TITAN_MASS_H
