#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "pyspring.h"

void bind_spring(py::module &m){
    py::class_<pySpring>(m, "Spring")
            .def(py::init<>(), py::return_value_policy::reference)
            .def(py::init<Spring *>(), py::return_value_policy::reference)
            .def("setK", &pySpring::setK, py::return_value_policy::reference)

            .def("_type", (int  (pySpring::*)()) &pySpring::_type, py::return_value_policy::reference)
            .def("_type", (void  (pySpring::*)(int)) &pySpring::_type, py::return_value_policy::reference)
            .def("_omega", (double  (pySpring::*)()) &pySpring::_omega, py::return_value_policy::reference)
            .def("_omega", (void  (pySpring::*)(double)) &pySpring::_omega, py::return_value_policy::reference)
            

            .def("setRestLength", &pySpring::setRestLength, py::return_value_policy::reference)
            .def("defaultLength", &pySpring::defaultLength, py::return_value_policy::reference)
            .def("setLeft", &pySpring::setLeft, py::return_value_policy::reference)
            .def("setRight", &pySpring::setRight, py::return_value_policy::reference)
            .def("setMasses", &pySpring::setMasses, py::return_value_policy::reference);
}