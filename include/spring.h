//
// Created by Jacob Austin on 5/17/18.
//
#ifndef LOCH_SPRING_H
#define LOCH_SPRING_H

#include "mass.h"
#include "vec.h"

class Mass;
struct CUDA_SPRING;
struct CUDA_MASS;

//Struct with CPU Spring properties used for optimal memory allocation on GPU memory

class Spring {
public:

    //Properties
    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)

    //Set
    Spring() { _left = nullptr; _right = nullptr; arrayptr = nullptr; _k = 10000.0; _rest = 1.0; }; //Constructor
    Spring(const CUDA_SPRING & spr);

    Spring(Mass * left, Mass * right, double k = 10000.0, double rest_len = 1.0) :
            _k(k), _rest(rest_len), _left(left), _right(right), arrayptr(nullptr) {}; //

    Spring(double k, double rest_length, Mass * left, Mass * right) :
            _k(k), _rest(rest_length), _left(left), _right(right) {};

    void setForce(); // w
    void setK(double k) { _k = k; } //sets K
    void setRestLength(double rest_length) { _rest = rest_length; } //sets Rest length
    void defaultLength(); //sets Rest Lenght

    void setLeft(Mass * left); // sets left mass (attaches spring to mass 1)
    void setRight(Mass * right);

    void setMasses(Mass * left, Mass * right) { _left = left; _right = right; } //sets both right and left masses

    //Get
    Vec getForce(); // computes force on right object. left force is - right force.

private:
    Mass * _left; // pointer to left mass object // private
    Mass * _right; // pointer to right mass object
    CUDA_SPRING *arrayptr; //Pointer to struct version for GPU cudaMalloc

    friend class Simulation;
    friend struct CUDA_SPRING;
    friend class Container;
    friend class Lattice;
    friend class Cube;
};
class pySpring {
public:
    pySpring() = default;
    pySpring(Spring * spring) { pointer = spring;};

    //Pointer to real C++ spring object
    Spring * pointer;

    //Properties
    double _k (){return pointer -> _k;} // spring constant (N/m)
    double _rest () {return pointer -> _rest;} // spring rest length (meters)

    //Set

    void pysetK(double k) { pointer -> _k = k; } //sets K
    void pysetRestLength(double rest_length) {pointer -> _rest = rest_length; } //sets Rest length
    void pydefaultLength() { pointer -> defaultLength();} //sets Rest Lenght

    void pysetLeft(Mass * left) {pointer -> setLeft(left);}; // sets left mass (attaches spring to mass 1)
    void pysetRight(Mass * right) {pointer -> setRight(right);};

    void pysetMasses(Mass * left, Mass * right) { pointer -> setLeft(left); pointer -> setRight(right); } //sets both right and left masses
};
struct CUDA_SPRING {
    CUDA_SPRING() {};
    CUDA_SPRING(const Spring & s);

    CUDA_SPRING(const Spring & s, CUDA_MASS * left, CUDA_MASS * right);

    CUDA_MASS * _left; // pointer to left mass object
    CUDA_MASS * _right; // pointer to right mass object

    double _k; // spring constant (N/m)
    double _rest; // spring rest length (meters)
};

#endif //LOCH_SPRING_H