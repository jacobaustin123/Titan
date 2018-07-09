//
// Created by Jacob Austin on 5/17/18.
//

#include "spring.h"

Vec Spring::getForce() { // computes force on right object. left force is - right force.
    Vec temp = (_right -> pos) - (_left -> pos);
    return _k * (_rest - temp.norm()) * (temp / temp.norm());
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
}

CUDA_SPRING::CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right) {
    _left = left;
    _right = right;
    _k = s._k;
    _rest = s._rest;
}