#ifdef _OPENMP
    #include "trift_omp.cc"
#else
    #include "trift_basic.cc"
#endif
