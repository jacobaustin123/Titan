#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "spring.h"

void bind_spring(py::module &m){
    py::class_<Spring>(m, "Spring")
            .def(py::init<>())
            .def("setK", &Spring::setK)
            .def("setRestLength", &Spring::setRestLength)
            .def("defaultLength", &Spring::defaultLength)
            .def("setLeft", &Spring::setLeft)
            .def("setRight", &Spring::setRight)
            .def("setMasses", &Spring::setMasses);

//            .def_readwrite("pointer", &pySpring::pointer)
//            .def("setK", &pySpring::pysetK)
//            .def("setRestLength", &pySpring::pysetRestLength)
//            .def("defaultLength", &pySpring::pydefaultLength)
//            .def("setLeft", &pySpring::pysetLeft)
//            .def("setRight", &pySpring::pysetRight)
//            .def("setMasses", &pySpring::pysetMasses);
}