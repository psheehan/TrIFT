#ifndef TRIFT_H
#define TRIFT_H

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <vector>
#include <complex>
#include "vector.h"

namespace py = pybind11;

const double pi = 3.14159265;

py::array_t<std::complex<double>> trift(py::array_t<double> x, 
        py::array_t<double> y, py::array_t<double> flux, py::array_t<double> u, 
        py::array_t<double> v, double dx, double dy, int nthreads);

py::array_t<std::complex<double>> trift_extended(py::array_t<double> x, 
        py::array_t<double> y, py::array_t<double> flux, py::array_t<double> u, 
        py::array_t<double> v, double dx, double dy, int nthreads);

py::array_t<std::complex<double>> trift2D(py::array_t<double> x, 
        py::array_t<double> y, py::array_t<double> flux, py::array_t<double> u, 
        py::array_t<double> v, double dx, double dy, int nthreads);

py::array_t<std::complex<double>> trift2D_extended(py::array_t<double> x, 
        py::array_t<double> y, py::array_t<double> flux, py::array_t<double> u, 
        py::array_t<double> v, double dx, double dy, int nthreads);

#endif
