//
// Created by rcorr on 9/12/2018.
//


#ifndef LOCH_PYMASS_H
#define LOCH_PYMASS_H

#include "mass.h"

class pyMass { // user mass class used to store real mass class properties in python and avoid ownership issues
public:
    //constructor
    pyMass() = default;
    pyMass(Mass * massp){ pointer = massp;}

    //Pointer to real C++ mass object
    Mass * pointer;

    //Properties
    double m() { return (pointer -> m);}  // mass in kg
    void m(double m) {pointer -> m = m;} // set mass
    double dt() { return (pointer -> dt);} // update interval
    double T() { return (pointer -> T);} // local time
    Vec pos() { return (pointer -> pos);} // position in m
    Vec vel() { return (pointer -> vel);} // velocity in m/s
    Vec acc() { return (pointer -> acc);} // acceleration in m/s^2
    Vec force() { return (pointer -> force);} // force in kg m / s^2

};

#endif //LOCH_PYMASS_H
