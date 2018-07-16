#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "spring.h"

void bind_sim(py::module &){
    py::class_<Spring>(m, "Spring")
            .def("setForce", &Spring::setForce)
            .def("setK", &Spring::setK)
            .def("setRestLength", &Spring::setRestLength)
            .def("defaultLength", &Spring::defaultLength)
            .def("setLeft", &Spring::setLeft)
            .def("setRight", &Spring::setRight)
            .def("setMasses", &Spring::setMasses)
            .def("getForce", &Spring::getForce);
}