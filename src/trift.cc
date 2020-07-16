#ifdef __CUDACC__
    #include "trift_cuda.cc"
#else
    #include "trift_cpu.cc"
#endif
