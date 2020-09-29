#ifndef TRIFT_H
#define TRIFT_H

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <vector>
#include <complex>

#ifdef __CUDACC__
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#endif

#include "vector.h"

namespace py = pybind11;

const double pi = 3.14159265;

py::array_t<std::complex<double>> trift(py::array_t<double> x, 
        py::array_t<double> y, py::array_t<double> flux, py::array_t<double> u, 
        py::array_t<double> v, double dx, double dy, int nthreads, 
        std::string mode);

#endif
