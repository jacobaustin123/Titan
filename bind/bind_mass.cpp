#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "mass.h"

void bind_mass(py::module &m){
    py::class_<Mass>(m, "Mass")
            .def(py::init<>())

            //properties
//            .def_readwrite("pointer", &pyMass::pointer)
//            .def("m", &pyMass::m)
//            .def("pos", &pyMass::pos)
//            .def("vel", &pyMass::vel)
//            .def("acc", &pyMass::acc)
//            .def("force", &pyMass::force)
//            .def("dt", &pyMass::dt)
//            .def("T", &pyMass::T);


//
            .def_readwrite("m", &Mass::m)
            .def_readwrite("pos", &Mass::pos)
            .def_readwrite("vel", &Mass::vel)
            .def_readwrite("acc", &Mass::acc)
            .def_readwrite("force", &Mass::force)
            .def_readwrite("dt", &Mass::dt)
            .def_readwrite("T", &Mass::T);

//Methods (Legacy)
//            .def("setMass", &Mass::setMass)
//            .def("setPos", &Mass::setPos)
//            .def("setVel", &Mass::setVel)
//            .def("setAcc", &Mass::setAcc)
//            .def("setForce", &Mass::setForce)
//            .def("setDeltaT", &Mass::setDeltaT)
//            .def("translate", &Mass::translate)
//            .def("makeFixed", &Mass::makeFixed)
//            .def("makeMovable", &Mass::makeMovable)
//            .def("isFixed", &Mass::isFixed)
//            .def("getMass", &Mass::getMass)
//            .def("getPosition", &Mass::getPosition)
//            .def("getVelocity", &Mass::getVelocity)
//            .def("getAcceleration", &Mass::getAcceleration)
//            .def("getForce", &Mass::getForce)
//            .def("time", &Mass::time)
//            .def("deltat", &Mass::deltat)
//            .def("stepTime", &Mass::stepTime);
}