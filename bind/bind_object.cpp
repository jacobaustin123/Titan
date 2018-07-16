#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "object.h"

void bind_sim(py::module &) {
    py::class_<Base>(m,"Base")
            .def(py::init<const std::string &>());

    // Method 2: pass parent class_ object:
    py::class_<Constraint>(m, "Constraint", Base /* <- specify Python parent type */)
            .def("getForce", &Constraint::getForce());

    py::class_<Ball>(m, "Ball", Constraint /* <- specify Python parent type */)
            .def("setMassValue", &Ball::setMassValue())
            .def("setKValue", &Ball::setKValue())
            .def("setDeltaTValue", &Ball::setDeltaTValue())
            .def("setRestLengthValue", &Ball::setRestLengthValue())
            .def("makeFixed", &Ball::makeFixed());

    //PLANE
    //CUBE
    //LATTICE
    //
}