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

