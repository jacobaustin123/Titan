//
// Created by Jacob Austin on 5/17/18.
//
#define GLM_FORCE_PURE
#include "spring.h"
#include <cmath>

const double EDGE_DAMPING = 20; // f_damp = delta_v_along_spring*edge_damping_constant;

Vec Spring::getForce() { // computes force on right object. left force is - right force.
  //    Vec temp = (_right -> pos) - (_left -> pos);
  //    return _k * (_rest - temp.norm()) * (temp / temp.norm());

    Vec temp = (_left -> pos) - (_right -> pos);
    Vec spring_force = _k * (temp.norm() - _rest) * (temp / temp.norm());

    spring_force += dot( (_left->vel - _right->vel) , temp/temp.norm() )*EDGE_DAMPING* (temp/temp.norm());
    return spring_force;
}

void Spring::setForce() { // computes force on right object. left force is - right force.
    Vec f = getForce();
    _right -> force += f;
    _left -> force += -f;
}

Spring::Spring(const CUDA_SPRING & spr) {
    this -> _k = spr._k;
    this -> _rest = spr._rest;
}

void Spring::defaultLength() { _rest = (_left -> pos - _right -> pos).norm() ; } //sets Rest Lenght

void Spring::setLeft(Mass * left) {
    if (_left) {
        _left -> decrementRefCount();
    }

    _left = left;
    _left -> ref_count++;

} // sets left mass (attaches spring to mass 1)

void Spring::setRight(Mass * right) {
    if (_right) {
        _right -> decrementRefCount();
    }

    _right = right;
    _right -> ref_count++;
}

CUDA_SPRING::CUDA_SPRING(const Spring & s) {
    _left = (s._left == nullptr) ? nullptr : s._left -> arrayptr;
    _right = (s._right == nullptr) ? nullptr : s. _right -> arrayptr;
    _k = s._k;
    _rest = s._rest;
    _type = s._type;
    _omega = s._omega;
}

CUDA_SPRING::CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right) {
    _left = left;
    _right = right;
    _k = s._k;
    _rest = s._rest;
    _type = s._type;
    _omega = s._omega;
}
