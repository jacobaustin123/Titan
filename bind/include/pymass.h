#ifndef LOCH_PYMASS_H
#define LOCH_PYMASS_H


#include "mass.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class pyMass { // user mass class used to store real mass class properties in python and avoid ownership issues
public:
    //constructor
    pyMass() = default;
    pyMass(Mass * massp){ pointer = massp;}

    //Pointer to real C++ mass object
    Mass * pointer;
    //Properties
    double m() { return (pointer -> m);}  // get mass [kg]
    void m(double m) {pointer -> m = m;} // set mass [kg]

    double dt() { return (pointer -> dt);} // get interval [s]
    void dt(double deltat) {pointer -> dt = deltat;} //set interval [s]

    double T() { return (pointer -> T);} // get local time [s]
    void T(double time){pointer -> T = time;} //set local time [s]

    double damping() {return pointer -> damping;} //get damping coefficient
    void damping(double damping){pointer-> damping = damping;} //set damping coefficient


    py::array_t<double> pos(); // get position [m]
    void pos(py::array_t<double> arr); // set position [m]
    py::array_t<double> vel(); //get velocity [m/s]
    void vel(py::array_t<double> arr); //set velocity [m/s]
    py::array_t<double> acc(); //get acceleration [m/s^2]
    void acc(py::array_t<double> arr); //set acceleration [m/s^2]
    py::array_t<double> force(); //get force [N]
    void force(py::array_t<double> arr); //set force [N]

};

#endif //LOCH_PYMASS_H
