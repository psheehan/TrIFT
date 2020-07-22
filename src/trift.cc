#ifdef __CUDACC__
    #include "trift_cuda.cc"
#else
    #include "trift_cpu.cc"
#endif

PYBIND11_MODULE(trift, m) {
    m.def("trift", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads) {
            py::gil_scoped_release release;
            return trift(x, y, flux, u, v, dx, dy, nthreads);},
            "Basic triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);

    m.def("trift_extended", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads) {
            py::gil_scoped_release release;
            return trift_extended(x, y, flux, u, v, dx, dy, nthreads);},
            "Extended triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);

    m.def("trift2D", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads) {
            py::gil_scoped_release release;
            return trift2D(x, y, flux, u, v, dx, dy, nthreads);},
            "Basic triangulated Fourier Transform of an image cube.",
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);

    m.def("trift2D_extended", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads) {
            py::gil_scoped_release release;
            return trift2D_extended(x, y, flux, u, v, dx, dy, nthreads);},
            "Extended triangulated Fourier Transform of an image cube.",
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);
}
