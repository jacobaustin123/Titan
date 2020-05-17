//
// Created by Jacob Austin on 5/17/18.
//
#include "spring.h"
#include <cmath>

namespace titan {


// this function is currently unneeded because springs do not need to be updated.
// this should be updated if any features are implemented to change spring parameters.
void Spring::update(const CUDA_SPRING & spr) {}

void Spring::defaultLength() { _rest = (_left -> pos - _right -> pos).norm() ; } // sets rest length

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
    _damping = s._damping;
}

CUDA_SPRING::CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right) {
    _left = left;
    _right = right;
    _k = s._k;
    _rest = s._rest;
    _type = s._type;
    _omega = s._omega;
    _damping = s._damping;
}

} // namespace titan