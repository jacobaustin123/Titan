//Include Python and pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <string>


//Declare Bind functions
void bind_mass(py::module &);
void bind_spring(py::module &);
void bind_sim(py::module &);
//void bind_object(py::module &);
//void bind_vec(py::module &);

//Create Module
PYBIND11_MODULE(bindloch, m) {
    bind_mass(m);
    bind_spring(m);
    bind_sim(m);
//    bind_object(m);
//    bind_vec(m);
}