//-------------------------------------------
//   Project: PLoch
//   Description: pymass implementation file
//-------------------------------------------


#include "pymass.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::array_t<double> pos() {

    auto result = py::array_t<double>(sizeof(double)* pointer -> pos);
    auto result_buffer = result.request();
    double *result_ptr = (double*) result_buffer.ptr;
    std::memcpy(result_ptr,result_vec.data(), result_vec.size() *sizeof(double));
}

void m(double m) {pointer -> m = m;} // set mass


