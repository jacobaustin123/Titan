//Include Python and pybind11
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <string>

//Include CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_device_runtime_api.h>


//Declare Bind functions
void bind_mass(py::module &);
void bind_spring(py::module &);
void bind_sim(py::module &);
void bind_object(py::module &);
void bind_vec(py::module &);

//Create Module
PYBIND11_MODULE(sim, m) {
    bind_mass(m);
    bind_spring(m);
    bind_sim(m);
    bind_object(m);
    bind_vec(m);
}