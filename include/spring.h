//
// Created by Jacob Austin on 5/17/18.
//

#ifndef LOCH_SPRING_H
#define LOCH_SPRING_H

#include "mass.h"
#include "vec.h"

struct CUDA_SPRING;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory

class Spring {
public:

    //Properties
    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)
    Mass * _left; // pointer to left mass object
    Mass * _right; // pointer to right mass object
    CUDA_SPRING *arrayptr; //Pointer to struct version for GPU cudaMalloc

    //Set
    Spring() { _left = nullptr; _right = nullptr; arrayptr = nullptr; _k = 10000.0; _rest = 1.0; }; //Constructor
    Spring(const CUDA_SPRING & spr); // Constructor

    Spring(Mass * left, Mass * right, double k = 10000.0, double rest_len = 1.0) :
            _k(k), _rest(rest_len), _left(left), _right(right), arrayptr(nullptr) {}; //

    Spring(double k, double rest_length, Mass * left, Mass * right) :
            _k(k), _rest(rest_length), _left(left), _right(right) {};

    void setForce(); // w
    void setK(double k) { _k = k; } //sets K
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength() { _rest = (_left -> pos - _right -> pos).norm() ; } //sets Rest Lenght
    void setLeft(Mass * left) { _left = left; } // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right) { _right = right; } //sets right mass (attaches spring to mass 2)
    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; } //sets both right and left masses

    //Get
    Vec getForce(); // computes force on right object. left force is - right force.
};

struct CUDA_SPRING {
    CUDA_SPRING() {};
    CUDA_SPRING(const Spring & s) {
        _left = (s._left == nullptr) ? nullptr : s._left -> arrayptr;
        _right = (s._right == nullptr) ? nullptr : s. _right -> arrayptr;
        _k = s._k;
        _rest = s._rest;
    }

    CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right) {
        _left = left;
        _right = right;
        _k = s._k;
        _rest = s._rest;
    }

    CUDA_MASS * _left; // pointer to left mass object
    CUDA_MASS * _right; // pointer to right mass object

    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)
};

#endif //LOCH_SPRING_H