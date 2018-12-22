
#include "pymass.h"
#include "pyspring.h"
#include "pysimulation.h"

pyMass pySimulation::createMass(py::array_t<double> arr){

    pyMass pm(sim.createMass(Vec (* arr.data(0), * arr.data(1), * arr.data(2))));
    return pm;
}

void pySimulation::createPlane(py::array_t<double> arr, double d) {

    Vec abc (* arr.data(0), * arr.data(1), * arr.data(2));
    sim.createPlane(Vec (* arr.data(0), * arr.data(1), * arr.data(2)), d);
}

void pySimulation::createBall(py::array_t<double> arr, double r ) {
    Vec abc (* arr.data(0), * arr.data(1), * arr.data(2));
    sim.createBall(Vec (* arr.data(0), * arr.data(1), * arr.data(2)), r);
}

