#ifdef __CUDACC__
    #include "trift_cuda.cc"
#else
    #include "trift_cpu.cc"
#endif

PYBIND11_MODULE(trift, m) {
    m.def("trift", &trift, "Basic triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);
    m.def("trift_extended", &trift_extended, 
            "Extended triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);
    m.def("trift2D", &trift2D, 
            "Basic triangulated Fourier Transform of an image cube.",
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);
    m.def("trift2D_extended", &trift2D_extended, 
            "Extended triangulated Fourier Transform of an image cube.",
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1);
}
