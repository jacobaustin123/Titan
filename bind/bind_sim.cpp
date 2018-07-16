#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "sim.h"

void bind_sim(py::module &){
    py::class_<Simulation>(m, "Sim")

            //Creators/Destructors
            .def("createMass", (void (Simulation::*)()) &Simulation::createMass)
            .def("createMass", (void (Simulation::*)(const Vec & pos)) &Simulation::createMass)
            .def("createSpring", (void (Simulation::*)()) &Simulation::createSpring)
            .def("createSpring", (void (Simulation::*)(Mass * m1, Mass * m2)) &Simulation::createSpring)
            .def("createPlane", &Simulation::createPlane)
            .def("createLattice", &Simulation::createLattice)

            .def("deleteMass", &Simulation::deleteMass)
            .def("deleteSpring", &Simulation::deleteSpring)
            .def("deleteContainer", &Simulation::deleteContainer)

                    //Setters
            .def("set", (void (Simulation::*)(Mass *m)) &Simulation::set)
            .def("set", (void (Simulation::*)(Spring *s)) &Simulation::set)
            .def("set", (void (Simulation::*)(Container *c)) &Simulation::set)
            .def("setAll", &Simulation::setAll)

            //Getters
            .def("get", (void (Simulation::*)(Mass *m)) &Simulation::get)
            .def("get", (void (Simulation::*)(Spring *s)) &Simulation::get)
            .def("get", (void (Simulation::*)(Container *c)) &Simulation::get)

                    //Bulk
            .def("setSpringConstant", &Simulation::setSpringConstant)
            .def("setMass", &Simulation::setMass)
            .def("setMassDeltaT", &Simulation::setMassDeltaT)
            .def("setAll", &Simulation::setAll)
            .def("getAll", &Simulation::getAll)

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