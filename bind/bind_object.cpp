#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "object.h"

void bind_sim(py::module &m) {
    // Method 2: pass parent class_ object:
    py::class_<Container>(m, "Container")
            .def("getForce", &Container::getForce());

    py::class_<Cube>(m, "Cube", Container /* <- specify Python parent type */)
            .def("setMassValue", &Ball::setMassValue())
            .def("setKValue", &Ball::setKValue())
            .def("setDeltaTValue", &Ball::setDeltaTValue())
            .def("setRestLengthValue", &Ball::setRestLengthValue())
            .def("makeFixed", &Ball::makeFixed());

    py::class_<Lattice>(m, "Lattice", Container /* <- specify Python parent type */)
            .def("setMassValue", &Lattice::setMassValue())
            .def("setKValue", &Lattice::setKValue())
            .def("setDeltaTValue", &Lattice::setDeltaTValue())
            .def("setRestLengthValue", &Lattice::setRestLengthValue())
            .def("makeFixed", &Lattice::makeFixed());

    //PLANE
    //CUBE
    //LATTICE
    //
}