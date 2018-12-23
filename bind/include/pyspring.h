//
// Created by rcorr on 9/12/2018.
//


#ifndef LOCH_PYSPRING_H
#define LOCH_PYSPRING_H

#include "spring.h"
#include "pymass.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class pySpring {
public:
    pySpring() = default;
    pySpring(Spring * spring) { pointer = spring;};
    //Pointer to real C++ spring object
    Spring * pointer;

    //Properties
    double k (){return pointer -> _k;} // spring constant (N/m)
    double rest () {return pointer -> _rest;} // spring rest length (meters)

    //BREATHING
    int _type() {return pointer -> _type;} // get  type 0-3
    void _type(int _type){pointer -> _type = _type} //set type
    double _omega() {return pointer -> _omega;} // get frequency
    void _omega(double _omega) { pointer -> _omega = _omega} // set frequency

    //Set

    void setK(double k) { pointer -> _k = k; } //sets K
    void setRestLength(double rest_length) {pointer -> setRestLength(rest_length); } //sets Rest length
    void defaultLength() { pointer -> defaultLength();} //sets Rest Lenght

    void setLeft(pyMass left) {pointer -> setLeft(left.pointer);}; // sets left mass (attaches spring to mass 1)
    void setRight(pyMass right) {pointer -> setRight(right.pointer);};
    void setMasses(pyMass left, pyMass right) { pointer -> setLeft(left.pointer); pointer -> setRight(right.pointer); } //sets both right and left masses
};

#endif //LOCH_PYSPRING_H
