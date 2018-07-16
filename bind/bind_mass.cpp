#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "mass.h"

void bind_sim(py::module &){
    py::class_<Mass>(m, "Mass")

            //Check that the constructors work
            .def(py:init<>())
            .def(py::init<(void (Mass::*)(struct CUDA_MASS &mass))>())
            .def(py:init<const Vec & position, double mass = 0.1, bool fixed = false, double dt = 0.0001>())


            .def("setMass", &Mass::setMass)
            .def("setPos", &Mass::setPos)
            .def("setVel", &Mass::setVel)
            .def("setAcc", &Mass::setAcc)
            .def("setForce", &Mass::setForce)
            .def("setDeltaT", &Mass::setDeltaT)
            .def("translate", &Mass::translate)
            .def("makeFixed", &Mass::makeFixed)
            .def("makeMovable", &Mass::makeMovable)
            .def("isFixed", &Mass::isFixed)
            .def("getMass", &Mass::getMass)
            .def("getPosition", &Mass::getPosition)
            .def("getVelocity", &Mass::getVelocity)
            .def("getAcceleration", &Mass::getAcceleration)
            .def("getForce", &Mass::getForce)
            .def("time", &Mass::time)
            .def("deltat", &Mass::deltat)
            .def("stepTime", &Mass::stepTime);
}