//-------------------------------------------
//   Project: PLoch
//   Description: pymass implementation file
//-------------------------------------------


#include "pymass.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

py::array_t<double> pyMass::pos() {

    auto pypos = py::array_t<double>(3);
    auto pypos_buffer = pypos.request();
    double *pypos_ptr = (double *) pypos_buffer.ptr;
    std::memcpy(pypos_ptr, &pointer->pos.data, 3 * sizeof(double));
    return pypos;
}

void pyMass::pos(py::array_t<double> arr){
    for (int i = 0; i< 3; i++){
        pointer -> pos[i] = * arr.data(i);
    }
}

py::array_t<double> pyMass::vel() {

    auto pyvel = py::array_t<double>(sizeof(double) * 3);
    auto pyvel_buffer = pyvel.request();
    double *pyvel_ptr = (double *) pyvel_buffer.ptr;
    std::memcpy(pyvel_ptr, &pointer->pos.data, 3 * sizeof(double));
    return pyvel;
}
void pyMass::vel(py::array_t<double> arr){
    for (int i = 0; i< 3; i++){
        pointer -> vel[i] = * arr.data(i);
    }
}

py::array_t<double> pyMass::acc() {

    auto pyacc = py::array_t<double>(sizeof(double) * 3);
    auto pyacc_buffer = pyacc.request();
    double *pyacc_ptr = (double *) pyacc_buffer.ptr;
    std::memcpy(pyacc_ptr, &pointer->pos.data, 3 * sizeof(double));
    return pyacc;
}

void pyMass::acc(py::array_t<double> arr){
    for (int i = 0; i< 3; i++){
        pointer -> acc[i] = * arr.data(i);
    }
}

py::array_t<double> pyMass::force() {

    auto pyforce = py::array_t<double>(sizeof(double) * 3);
    auto pyforce_buffer = pyforce.request();
    double *pyforce_ptr = (double *) pyforce_buffer.ptr;
    std::memcpy(pyforce_ptr, &pointer->pos.data, 3 * sizeof(double));
    return pyforce;
}

void pyMass::force(py::array_t<double> arr){
    for (int i = 0; i< 3; i++){
        pointer -> force[i] = * arr.data(i);
    }
}