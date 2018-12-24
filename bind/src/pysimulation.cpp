
#include "pymass.h"
#include "pyspring.h"
#include "pysimulation.h"

pyMass pySimulation::createMass(py::array_t<double> arr){

    pyMass pm(sim.createMass(Vec (* arr.data(0), * arr.data(1), * arr.data(2))));
    return pm;
}

void pySimulation::createCube(py::array_t<double> center, double side_lenght)
{

    Vec centervec (* center.data(0), * center.data(1), * center.data(2));
    sim.createCube(centervec, side_lenght);

}
void pySimulation::createLattice(py::array_t<double> center, py::array_t<double> dims, int nx, int ny, int nz){

    Vec centervec (* center.data(0), * center.data(1), * center.data(2));
    Vec dimsvec (* dims.data(0), * dims.data(1), * dims.data(2));
    sim.createLattice(centervec, dimsvec, nx, ny, nz);
}

//void pySimulation::createRobot(py::array_t<double> center, double side_lenght){
//
//    Vec centervec (* center.data(0), * center.data(1), * center.data(2));
//    sim.createRobot(centervec, side_lenght);
//}

void pySimulation::createBeam(py::array_t<double> center, py::array_t<double> dims, int nx, int ny, int nz){
    Vec centervec (* center.data(0), * center.data(1), * center.data(2));
    Vec dimsvec (* dims.data(0), * dims.data(1), * dims.data(2));
    sim.createBeam(centervec, dimsvec, nx, ny, nz);
}


void pySimulation::createPlane(py::array_t<double> arr, double d) {

    sim.createPlane(Vec (* arr.data(0), * arr.data(1), * arr.data(2)), d);
}

void pySimulation::createBall(py::array_t<double> arr, double r ) {
    Vec abc (* arr.data(0), * arr.data(1), * arr.data(2));
    sim.createBall(Vec (* arr.data(0), * arr.data(1), * arr.data(2)), r);
}

