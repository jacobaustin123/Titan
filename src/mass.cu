//
// Created by Jacob Austin on 5/17/18.
//

#include "mass.h"

Mass::Mass() {
    m = 1.0;
    dt = 0.0001;
    T = 0;
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
} // constructor TODO fix timing

void Mass::operator=(CUDA_MASS & mass) {
    m = mass.m;
    dt = mass.dt;
    T = mass.T;
    pos = mass.pos;
    vel = mass.vel;
    acc = mass.acc;
    force = mass.force;
    valid = mass.valid;

    ref_count = this -> ref_count;
    arrayptr = this -> arrayptr;

#ifdef CONSTRAINTS
    constraints = this -> constraints;
#endif

#ifdef GRAPHICS
    color = mass.color;
#endif
}

Mass::Mass(const Vec & position, double mass, bool fixed, double dt) {
    m = mass;
    pos = position;

    this -> dt = dt;

    T = 0;
    valid = true;
    arrayptr = nullptr;
    ref_count = 0;

#ifdef GRAPHICS
    color = Vec(1.0, 0.2, 0.2);
#endif
}

CUDA_MASS::CUDA_MASS(Mass &mass) {
    m = mass.m;
    dt = mass.dt;
    T = mass.T;
    pos = mass.pos;
    vel = mass.vel;
    acc = mass.acc;
    force = mass.force;
    valid = true;

#ifdef CONSTRAINTS
    constraints = CUDA_LOCAL_CONSTRAINTS(mass.constraints);
#endif

#ifdef GRAPHICS
    color = mass.color;
#endif
}

#ifdef CONSTRAINTS
void Mass::addConstraint(CONSTRAINT_TYPE type, const Vec & vec, double num) { // TODO make this more efficient
    if (type == 0) {
        this -> constraints.constraint_plane.push_back(CudaConstraintPlane(vec, num));
        this -> constraints.num_constraint_planes++;
        this -> constraints.constraint_plane_ptr = thrust::raw_pointer_cast(constraints.constraint_plane.data());
    } else if (type == 1) {
        this -> constraints.contact_plane.push_back(CudaContactPlane(vec, num));
        this -> constraints.num_contact_planes++;
        this -> constraints.contact_plane_ptr = thrust::raw_pointer_cast(constraints.contact_plane.data());
    } else if (type == 2) {
        this -> constraints.ball.push_back(CudaBall(vec, num));
        this -> constraints.num_balls++;
        this -> constraints.ball_ptr = thrust::raw_pointer_cast(constraints.ball.data());
    } else if (type == 3) {
        this -> constraints.direction.push_back(CudaDirection(vec, num));
        this -> constraints.num_directions++;
        this -> constraints.direction_ptr = thrust::raw_pointer_cast(constraints.direction.data());
    }
}

void Mass::fix() {
    this -> constraints.fixed = true;
}
void Mass::unfix() {
    this -> constraints.fixed = false;
}

#endif

void decrementRefCount(Mass * m) {
    if (--m -> ref_count == 0) {

        if (m -> arrayptr) {
            cudaFree(m -> arrayptr);
        }

        delete m;
    }
}