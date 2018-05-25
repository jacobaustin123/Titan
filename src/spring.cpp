//
// Created by Jacob Austin on 5/17/18.
//

#include "spring.h"

Vec Spring::getForce() { // computes force on right object. left force is - right force.
    Vec temp = (_right -> getPosition()) - (_left -> getPosition());
    return _k * (_rest - temp.norm()) * (temp / temp.norm());
}

void Spring::setForce() { // computes force on right object. left force is - right force.
    Vec f = getForce();
    _right -> addForce(f);
    _left -> addForce(-f);
}

