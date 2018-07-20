//#include <pybind11/pybind11.h>
//namespace py = pybind11;
//
//#include "object.h"
//#include "mass.h"
//#include "spring.h"
//
//void bind_object(py::module &m) {
//    // Method 2: pass parent class_ object:
//    py::class_<Container>(m, "Container")
//            .def(py::init<>())
//            .def("translate", &Container::translate)
//            .def("setMassValues", &Container::setMassValues)
//            .def("setSpringConstants", &Container::setSpringConstants)
//            .def("setDeltaT", &Container::setDeltaT)
//            .def("setRestLengths", &Container::setRestLengths)
//            .def("makeFixed", &Container::makeFixed)
//            .def("add", (void (Container::*)(Mass * m)) &Container::add)
//            .def("add", (void (Container::*)(Spring *s)) &Container::add)
//            .def("add", (void (Container::*)(Container *c)) &Container::add);
//
//
////    py::class_<Cube>(m, "Cube", Container /* <- specify Python parent type */)
////            .def(py::init<>())
////
////    py::class_<Lattice>(m, "Lattice", Container /* <- specify Python parent type */)
////            .def(py::init<>())
////
////
////    py::class_<Ball>(m, "Ball", Constraint /* <- specify Python parent type */)
////            .def(py::init<>());
////
////    py::class_<ContactPlane>(m, "ContactPlane", Constraint /* <- specify Python parent type */)
////            .def(py::init<>());
////
//
//
//    //PLANE
//    //CUBE
//    //LATTICE
//    //
//}