//-------------------------------------------
//   Project: PLoch
//   Description: pymass implementation file
//-------------------------------------------


#include "pymass.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::array_t<double> pyMass::pos() {

    auto pypos = py::array_t<double>(sizeof(double) * 3);
    auto pypos_buffer = pypos.request();
    double *pypos_ptr = (double *) pypos_buffer.ptr;
    std::memcpy(pypos_ptr, &pointer->pos.data, 3 * sizeof(double));

    return pypos;
}
