#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "pymass.h"

void bind_mass(py::module &m){
    py::class_<pyMass>(m, "Mass")
            .def(py::init<>(), py::return_value_policy::reference)

            //properties
            .def_readwrite("pointer", &pyMass::pointer)
//            .def("m", &pyMass::m)

            .def("m", (double (pyMass::*)()) &pyMass::m, py::return_value_policy::reference)
            .def("m", (void (pyMass::*)(double m)) &pyMass::m, py::return_value_policy::reference)
            .def("pos", (py::array_t<double>  (pyMass::*)()) &pyMass::pos, py::return_value_policy::reference)
            .def("pos", (void  (pyMass::*)(py::array_t<double> arr)) &pyMass::pos, py::return_value_policy::reference)
            .def("vel", &pyMass::vel)
            .def("acc", &pyMass::acc)
            .def("force", &pyMass::force)
            .def("dt", &pyMass::dt)
            .def("T", &pyMass::T);
}