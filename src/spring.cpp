//
// Created by Jacob Austin on 5/17/18.
//

#include "spring.h"

Vec Spring::getForce() { // computes force on right object. left force is - right force.
    Vec temp = (_right -> getPosition()) - (_left -> getPosition());
    return temp * _k * (_rest - temp.norm()) / temp.norm();
}

void Spring::setForce() { // computes force on right object. left force is - right force.
    Vec f = getForce();
//    std::cout << f << std::endl;
    _left -> setForce(-f);
    _right -> setForce(f);
}