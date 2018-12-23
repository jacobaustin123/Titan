#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

#include "pymass.h"

void bind_mass(py::module &m){
    py::class_<pyMass>(m, "Mass")
            .def(py::init<>(), py::return_value_policy::reference)

            //properties
            .def_readwrite("pointer", &pyMass::pointer)
            .def("m", (double (pyMass::*)()) &pyMass::m, py::return_value_policy::reference)
            .def("m", (void (pyMass::*)(double m)) &pyMass::m, py::return_value_policy::reference)
            .def("dt", (double  (pyMass::*)()) &pyMass::dt, py::return_value_policy::reference)
            .def("dt", (void  (pyMass::*)(double)) &pyMass::dt, py::return_value_policy::reference)
            .def("T", (double  (pyMass::*)()) &pyMass::T, py::return_value_policy::reference)
            .def("T", (void  (pyMass::*)(double)) &pyMass::T, py::return_value_policy::reference)

            .def("pos", (py::array_t<double>  (pyMass::*)()) &pyMass::pos, py::return_value_policy::reference)
            .def("pos", (void  (pyMass::*)(py::array_t<double> arr)) &pyMass::pos, py::return_value_policy::reference)
            .def("vel", (py::array_t<double>  (pyMass::*)()) &pyMass::vel, py::return_value_policy::reference)
            .def("vel", (void  (pyMass::*)(py::array_t<double> arr)) &pyMass::vel, py::return_value_policy::reference)
            .def("acc", (py::array_t<double>  (pyMass::*)()) &pyMass::acc, py::return_value_policy::reference)
            .def("acc", (void  (pyMass::*)(py::array_t<double> arr)) &pyMass::acc, py::return_value_policy::reference)
            .def("force", (py::array_t<double>  (pyMass::*)()) &pyMass::force, py::return_value_policy::reference)
            .def("force", (void  (pyMass::*)(py::array_t<double> arr)) &pyMass::force, py::return_value_policy::reference);

}