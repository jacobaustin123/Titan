#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "vec.h"
#include "sim.h"
#include "mass.h"
#include "spring.h"
#include "pymass.h"
#include "pysimulation.h"


void bind_sim(py::module &m){

    py::class_<pySimulation>(m, "Sim")
            .def(py::init<>(),py::return_value_policy::reference)
            .def("createMass", (pyMass (pySimulation::*)()) &pySimulation::createMass,
                 py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>(), py::return_value_policy::reference)

            .def("createMass", (pyMass (pySimulation::*)(py::array_t<double> arr)) &pySimulation::createMass,
                 py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>(), py::return_value_policy::reference)

            .def("createSpring", (pySpring (pySimulation::*)()) &pySimulation::createSpring, py::return_value_policy::reference)
            .def("createSpring", (pySpring (pySimulation::*)(pyMass m1, pyMass m2)) &pySimulation::createSpring, py::return_value_policy::reference)
            .def("createPlane", &pySimulation::createPlane, py::return_value_policy::reference)
            .def("createBall", &pySimulation::createBall, py::return_value_policy::reference)

//            .def("createLattice", &pySimulation::createLattice)

//            .def("deleteMass", &pySimulation::deleteMass)
//            .def("deleteSpring", &pySimulation::deleteSpring)
//            .def("deleteContainer", &pySimulation::deleteContainer)
//            .def("clearConstraints", &pySimulation::clearConstraints)
//                    //Setters
//            .def("set", (void (pySimulation::*)(Mass *m)) &pySimulation::set)
//            .def("set", (void (pySimulation::*)(Spring *s)) &pySimulation::set)
//            .def("set", (void (pySimulation::*)(Container *c)) &pySimulation::set)
            .def("setAll", &pySimulation::setAll)
//
//            //Getters
//            .def("get", (void (pySimulation::*)(Mass *m)) &pySimulation::get)
//            .def("get", (void (pySimulation::*)(Spring *s)) &pySimulation::get)
//            .def("get", (void (pySimulation::*)(Container *c)) &pySimulation::get)

                    //Bulk
            .def("setSpringConstant", &pySimulation::setSpringConstant)
            .def("setMass", &pySimulation::setMassValues)
            .def("setMassDeltaT", &pySimulation::setDeltaT)
            .def("setAll", &pySimulation::setAll)
            .def("getAll", &pySimulation::getAll)
            .def("defaultRestLength", &pySimulation::defaultRestLength)

                    //Control

            .def("start", &pySimulation::start,
                 py::call_guard<py::scoped_ostream_redirect,
                         py::scoped_estream_redirect>(),py::return_value_policy::reference)
            .def("stop", (void (pySimulation::*)()) &pySimulation::stop)
            .def("stop", (void (pySimulation::*)(double time)) &pySimulation::stop)
            .def("pause", &pySimulation::pause)
            .def("resume", &pySimulation::resume)
            .def("wait", &pySimulation::wait)
            .def("time", &pySimulation::time)
            .def("running", &pySimulation::running);
}