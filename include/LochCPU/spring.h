//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SPRING_H
#define LOCH_SPRING_H

#include "mass.h"
#include "vec.h"

class Spring {
public:
    Spring() {}; // constructors
    Spring(Mass * left, Mass * right, double K = 1.0, double rest_len = 1.0) : _k(K), _rest(rest_len), _left(left), _right(right) {};
    Spring(double k, double rest_length, Mass * left, Mass * right) : _k(k), _rest(rest_length), _left(left), _right(right) {};

    Vec getForce(); // computes force on right object. left force is - right force.
    void setForce(); // adds force to both right and left elements

    void setK(double k) { _k = k; }
    void setRestLength(double rest_length) { _rest = rest_length; }
    void setLeft(Mass * left) { _left = left; }
    void setRight(Mass * right) { _right = right; }
    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; }

    Mass * _left; // pointer to left mass object
    Mass * _right; // pointer to right mass object

    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)
};

#endif //LOCH_SPRING_H