#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "sim.h"

void bind_sim(py::module &m){
    py::class_<Simulation>(m, "Sim")
            .def(py::init<>())
            //Creators/Destructors
            .def("createMass", (Mass * (Simulation::*)()) &Simulation::createMass)
            .def("createMass", (Mass * (Simulation::*)(const Vec & pos)) &Simulation::createMass)
            .def("createSpring", (Spring * (Simulation::*)()) &Simulation::createSpring)
            .def("createSpring", (Spring * (Simulation::*)(Mass * m1, Mass * m2)) &Simulation::createSpring)
            .def("createPlane", &Simulation::createPlane)
            .def("createLattice", &Simulation::createLattice)

            .def("deleteMass", &Simulation::deleteMass)
            .def("deleteSpring", &Simulation::deleteSpring)
            .def("deleteContainer", &Simulation::deleteContainer)
            .def("clearConstraints", &Simulation::clearConstraints)
//                    //Setters
            .def("set", (void (Simulation::*)(Mass *m)) &Simulation::set)
            .def("set", (void (Simulation::*)(Spring *s)) &Simulation::set)
            .def("set", (void (Simulation::*)(Container *c)) &Simulation::set)
            .def("setAll", &Simulation::setAll)
//
//            //Getters
            .def("get", (void (Simulation::*)(Mass *m)) &Simulation::get)
            .def("get", (void (Simulation::*)(Spring *s)) &Simulation::get)
            .def("get", (void (Simulation::*)(Container *c)) &Simulation::get)

                    //Bulk
            .def("setSpringConstant", &Simulation::setSpringConstant)
            .def("setMass", &Simulation::setMassValues)
            .def("setMassDeltaT", &Simulation::setDeltaT)
            .def("setAll", &Simulation::setAll)
            .def("getAll", &Simulation::getAll)
            .def("defaultRestLength", &Simulation::defaultRestLength)

                    //Control

            .def("start", &Simulation::start)
            .def("stop", (void (Simulation::*)()) &Simulation::stop)
            .def("stop", (void (Simulation::*)(double time)) &Simulation::stop)
            .def("pause", &Simulation::pause)
            .def("resume", &Simulation::resume)
            .def("wait", &Simulation::wait)
            .def("time", &Simulation::time)
            .def("running", &Simulation::running);
}