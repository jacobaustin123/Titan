#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "vec.h"
#include "sim.h"
#include "mass.h"
#include "spring.h"
#include "pymass.h"


void bind_sim(py::module &m){

    py::class_<Simulation>(m, "Sim")
            .def(py::init<>())
            //Creators/Destructors
//            .def("createMass", (pyMass (Simulation::*)()) &Simulation::createMass,
//                 py::call_guard<py::scoped_ostream_redirect,
//                         py::scoped_estream_redirect>(), py::return_value_policy::reference)
        .def("createMass", [](Func&& f = (Mass * (Simulation::*)()) &Simulation::createMass){
                pyMass pm(f);
                return pm;
            })

//
//            .def("createMass", (pyMass (Simulation::*)(const Vec & pos)) &Simulation::createMass,
//                 py::call_guard<py::scoped_ostream_redirect,
//                         py::scoped_estream_redirect>(), py::return_value_policy::reference)
//            .def("createSpring", (pySpring (Simulation::*)()) &Simulation::createSpring)
//            .def("createSpring", (pySpring (Simulation::*)(pyMass m1, pyMass m2)) &Simulation::createSpring)
//            .def("createPlane", &Simulation::createPlane)
//            .def("createPlane", [](py::array_t<double> array, double d, void (Simulation::* createPlaneFunc) (const Vec &, double) = &Simulation::createPlane){
//                Vec array_vec;
//                std::memcpy(&array_vec[0], array.data(), array.size() *sizeof(double));
//                createPlaneFunc(array_vec, d);
//
//            }) NOT WORKING BECAUSE IT IS NOT POSSIBLE TO CALL A FUNCTION POINTER TO A METHOD WOTHOUR KNOWING THE
//            OBJECT IT IS APPLIED TO IMPLICITLY

//            .def("createLattice", &Simulation::createLattice)

            .def("deleteMass", &Simulation::deleteMass)
            .def("deleteSpring", &Simulation::deleteSpring)
//            .def("deleteContainer", &Simulation::deleteContainer)
//            .def("clearConstraints", &Simulation::clearConstraints)
//                    //Setters
            .def("set", (void (Simulation::*)(Mass *m)) &Simulation::set)
            .def("set", (void (Simulation::*)(Spring *s)) &Simulation::set)
//            .def("set", (void (Simulation::*)(Container *c)) &Simulation::set)
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