#ifdef _OPENMP
#include <omp.h>
#endif

#include "trift.h"
#include <delaunator-header-only.hpp>
#include "timer.h"
#include "fastmath.h"
#include <unordered_map>

void ft(double *x, double *y, double *flux, double *u, double *v, 
        std::complex<double> *vis, int nu, std::vector<long unsigned int> 
        triangles, int nthreads) {
    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    #ifdef _OPENMP
    std::complex<double> **vis_tmp = new std::complex<double>*[nthreads];
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        vis_tmp[i] = new std::complex<double>[nu];
    #endif

    #pragma omp parallel
    {
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();

    for (std::size_t i = 0; i < (std::size_t) nu; i++)
        vis_tmp[thread_id][i] = 0;
    #endif

    #pragma omp for
    for (std::size_t i = 0; i < triangles.size(); i+=3) {
        double intensity_triangle = (flux[triangles[i]] + 
            flux[triangles[i+1]] + flux[triangles[i+2]]) / 3.;

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln = rn1 - rn;
            Vector<double, 3> ln_1 = rn - rn_1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

                double rn_dot_uv = rn.dot(uv);
                
                #ifdef _OPENMP
                vis_tmp[thread_id][k] += intensity_triangle * 
                    ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                    (FastCos(rn_dot_uv) - I*FastSin(rn_dot_uv));
                #else
                vis[k] += intensity_triangle * ln_1_dot_zhat_cross_ln /
                    (ln.dot(uv) * ln_1.dot(uv)) * (FastCos(rn_dot_uv) - 
                    I*FastSin(rn_dot_uv));
                #endif
            }
        }
    }
    }

    // Now add together all of the separate vis'.

    #ifdef _OPENMP
    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        for (std::size_t j = 0; j < (std::size_t) nthreads; j++) {
            vis[i] += vis_tmp[j][i];
        }
    }

    // And clean up the tmp arrays.
    
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        delete[] vis_tmp[i];
    delete[] vis_tmp;
    #endif
}

void ft_extended(double *x, double *y, double *flux, double *u, double *v, 
        std::complex<double> *vis, int nu, std::vector<long unsigned int> 
        triangles, int nthreads) {
    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    #ifdef _OPENMP
    std::complex<double> **vis_tmp = new std::complex<double>*[nthreads];
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        vis_tmp[i] = new std::complex<double>[nu];
    #endif

    #pragma omp parallel
    {
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();

    for (std::size_t i = 0; i < (std::size_t) nu; i++)
        vis_tmp[thread_id][i] = 0;
    #endif

    Vector<std::complex<double>, 3> *integral_part1 = new 
            Vector<std::complex<double>,3>[nu];
    std::complex<double> *integral_part2 = new std::complex<double>[nu];

    #pragma omp for
    for (std::size_t i = 0; i < triangles.size(); i+=3) {
        // Calculate the area of the triangle.

        Vector<double, 3> Vertex1(x[triangles[i+0]], y[triangles[i+0]], 0.);
        Vector<double, 3> Vertex2(x[triangles[i+1]], y[triangles[i+1]], 0.);
        Vector<double, 3> Vertex3(x[triangles[i+2]], y[triangles[i+2]], 0.);

        Vector<double, 3> Side1 = Vertex2 - Vertex1;
        Vector<double, 3> Side2 = Vertex3 - Vertex1;

        double Area = 0.5 * (Side1.cross(Side2)).norm();

        // Precompute some aspects of the integral that remain the same.

        for (int m = 0; m < 3; m++) {
            // Get the appropriate vertices.

            std::size_t i_rm1 = triangles[i + (m+1)%3];
            Vector<double, 3> rm1(x[i_rm1], y[i_rm1],  0.);

            std::size_t i_rm = triangles[i + m];
            Vector<double, 3> rm(x[i_rm], y[i_rm],  0.);

            // Calculate the needed derivatives of those.

            Vector<double, 3> lm = rm1 - rm;
            Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
            Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

            // Now loop through the uv points and calculate the pieces of the 
            // integral.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = BesselJ0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                std::complex<double> exp_part = (FastCos(r_mc.dot(uv)) -  
                        I*FastSin(r_mc.dot(uv))) / (uv.dot(uv));

                // Now add everything together.

                integral_part1[k] += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2[k] += -I*bessel0_prefix3 * bessel0 * exp_part;
            }
        }

        // Now loop through an do the actual calculation.

        for (int j = 0; j < 3; j++) {
            double intensity = flux[triangles[i + j]];

            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln1 = rn_1 - rn1;

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            // Finally put into the real and imaginary components.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                #ifdef _OPENMP
                vis_tmp[thread_id][k] += intensity * zhat_cross_ln1.dot(
                        integral_part1[k]+rn1*integral_part2[k]) / (2.*Area);
                #else
                vis[k] += intensity * zhat_cross_ln1.dot(integral_part1[k] + 
                        rn1*integral_part2[k]) / (2.*Area);
                #endif
            }
        }

        // Clear out the integral array for the next triangle.

        for (std::size_t k = 0; k < (std::size_t) nu; k++) {
            integral_part1[k] = 0.;
            integral_part2[k] = 0.;
        }
    }

    // Clean up.

    delete[] integral_part1; delete [] integral_part2;
    }

    #ifdef _OPENMP
    // Now add together all of the separate vis'.

    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        for (std::size_t j = 0; j < (std::size_t) nthreads; j++) {
            vis[i] += vis_tmp[j][i];
        }
    }

    // And clean up the tmp arrays.
    
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        delete[] vis_tmp[i];
    delete[] vis_tmp;
    #endif
}

void ft2D(double *x, double *y, double *flux, double *u, double *v, 
        std::complex<double> *vis, int nu, int nv, 
        std::vector<long unsigned int> triangles, int nthreads) {
    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    #ifdef _OPENMP
    std::complex<double> **vis_tmp = new std::complex<double>*[nthreads];
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++) 
        vis_tmp[i] = new std::complex<double>[nu*nv];
    #endif

    #pragma omp parallel
    {
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();

    for (std::size_t i = 0; i < (std::size_t) nu*nv; i++)
        vis_tmp[thread_id][i] = 0;
    #endif

    double *intensity_triangle = new double[nv];

    #pragma omp for
    for (std::size_t i = 0; i < triangles.size(); i+=3) {
        // Get the intensity of the triangle at each wavelength.

        for (std::size_t l = 0; l < (std::size_t) nv; l++) {
            intensity_triangle[l] = (flux[triangles[i]*nv+l] + 
                flux[triangles[i+1]*nv+l] + flux[triangles[i+2]*nv+l]) / 3.;
        }

        // Calculate the FT

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln = rn1 - rn;
            Vector<double, 3> ln_1 = rn - rn_1;

            // Now loop through the UV points and calculate the Fourier
            // Transform.

            double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

                double rn_dot_uv = rn.dot(uv);
                
                std::size_t idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    #ifdef _OPENMP
                    vis_tmp[thread_id][idy+l] += intensity_triangle[l] * 
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                        (FastCos(rn_dot_uv) - I*FastSin(rn_dot_uv));
                    #else
                    vis[idy+l] += intensity_triangle[l] *
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) *
                        (FastCos(rn_dot_uv) - I*FastSin(rn_dot_uv));
                    #endif
                }
            }
        }
    }
    delete[] intensity_triangle;
    }

    // Now add together all of the separate vis'.

    #ifdef _OPENMP
    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t) nu*nv; i++) {
        for (std::size_t j = 0; j < (std::size_t) nthreads; j++) {
            vis[i] += vis_tmp[j][i];
        }
    }

    // And clean up the tmp arrays.
    
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        delete[] vis_tmp[i];
    delete[] vis_tmp;
    #endif
}

void ft2D_extended(double *x, double *y, double *flux, double *u, double *v, 
        std::complex<double> *vis, int nu, int nv, 
        std::vector<long unsigned int> triangles, int nthreads) {
    // Loop through and take the Fourier transform of each triangle.
    
    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    #ifdef _OPENMP
    std::complex<double> **vis_tmp = new std::complex<double>*[nthreads];
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        vis_tmp[i] = new std::complex<double>[nu*nv];
    #endif

    #pragma omp parallel
    {
    #ifdef _OPENMP
    int thread_id = omp_get_thread_num();

    for (std::size_t i = 0; i < (std::size_t) nu*nv; i++)
        vis_tmp[thread_id][i] = 0;
    #endif

    Vector<std::complex<double>, 3> *integral_part1 = new 
            Vector<std::complex<double>,3>[nu];
    std::complex<double> *integral_part2 = new std::complex<double>[nu];

    #pragma omp for
    for (std::size_t i = 0; i < triangles.size(); i+=3) {
        // Calculate the area of the triangle.

        Vector<double, 3> Vertex1(x[triangles[i+0]], y[triangles[i+0]], 0.);
        Vector<double, 3> Vertex2(x[triangles[i+1]], y[triangles[i+1]], 0.);
        Vector<double, 3> Vertex3(x[triangles[i+2]], y[triangles[i+2]], 0.);

        Vector<double, 3> Side1 = Vertex2 - Vertex1;
        Vector<double, 3> Side2 = Vertex3 - Vertex1;

        double Area = 0.5 * (Side1.cross(Side2)).norm();

        // Precompute some aspects of the integral that remain the same.

        for (int m = 0; m < 3; m++) {
            // Get the appropriate vertices.

            std::size_t i_rm1 = triangles[i + (m+1)%3];
            Vector<double, 3> rm1(x[i_rm1], y[i_rm1],  0.);

            std::size_t i_rm = triangles[i + m];
            Vector<double, 3> rm(x[i_rm], y[i_rm],  0.);

            // Calculate the needed derivatives of those.

            Vector<double, 3> lm = rm1 - rm;
            Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
            Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

            // Now loop through the uv points and calculate the pieces of the 
            // integral.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = BesselJ0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        BesselJ1(uv.dot(lm)/2.);

                std::complex<double> exp_part = (FastCos(r_mc.dot(uv)) -  
                        I*FastSin(r_mc.dot(uv))) / (uv.dot(uv));

                // Now add everything together.

                integral_part1[k] += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2[k] += -I*bessel0_prefix3 * bessel0 * exp_part;
            }
        }

        // Now loop through an do the actual calculation.

        for (int j = 0; j < 3; j++) {
            // Calculate the vectors for the vertices of the triangle.

            std::size_t i_rn1 = triangles[i + (j+1)%3];
            Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

            std::size_t i_rn = triangles[i + j];
            Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

            std::size_t i_rn_1 = triangles[i + (j+2)%3];
            Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

            // Calculate the vectors for the edges of the triangle.

            Vector<double, 3> ln1 = rn_1 - rn1;

            Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

            // Finally put into the real and imaginary components.

            for (std::size_t k = 0; k < (std::size_t) nu; k++) {
                std::size_t idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    #ifdef _OPENMP
                    vis_tmp[thread_id][idy+l] += flux[triangles[i+j]*
                            nv+l] * zhat_cross_ln1.dot(integral_part1[k]+rn1*
                            integral_part2[k]) / (2.*Area);
                    #else
                    vis[idy+l] += flux[triangles[i+j]*nv+l] *
                            zhat_cross_ln1.dot(integral_part1[k]+rn1*
                            integral_part2[k]) / (2.*Area);
                    #endif
                }
            }
        }

        // Clear out the integral arrays for the next triangle.

        for (std::size_t k = 0; k < (std::size_t) nu; k++) {
            integral_part1[k] = 0.;
            integral_part2[k] = 0.;
        }
    }
    // Clean up.

    delete[] integral_part1; delete [] integral_part2;
    }

    // Now add together all of the separate vis'.

    #ifdef _OPENMP
    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t) nu*nv; i++)
        for (std::size_t j = 0; j < (std::size_t) nthreads; j++)
            vis[i] += vis_tmp[j][i];

    // And clean up the tmp arrays.
    
    for (std::size_t i = 0; i < (std::size_t) nthreads; i++)
        delete[] vis_tmp[i];
    delete[] vis_tmp;
    #endif
}

py::array_t<std::complex<double>> trift(py::array_t<double> _x, 
        py::array_t<double> _y, py::array_t<double> _flux, 
        py::array_t<double> _u, py::array_t<double> _v, double dx, double dy, 
        int nthreads, std::string mode) {

    py::gil_scoped_acquire acquire;

    // Convert from Python to C++ useful things.
    
    auto x_buf = _x.request(); auto y_buf = _y.request(); 
    auto flux_buf = _flux.request(); auto u_buf = _u.request();
    auto v_buf = _v.request();

    if (x_buf.ndim != 1 || y_buf.ndim != 1 || flux_buf.ndim > 2 || 
            u_buf.ndim != 1 || v_buf.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");

    int nx = x_buf.shape[0]; int nu = u_buf.shape[0];
    int nv = 1;
    if (flux_buf.ndim == 2)
        nv = flux_buf.shape[1];

    double *x = (double *) x_buf.ptr,
           *y = (double *) y_buf.ptr,
           *flux = (double *) flux_buf.ptr,
           *u = (double *) u_buf.ptr,
           *v = (double *) v_buf.ptr;

    // Setup the resulting array.

    auto _vis = py::array_t<std::complex<double>>(nu);
    if (flux_buf.ndim == 2) _vis.resize({nu, nv});

    auto vis_buf = _vis.request();
    std::complex<double> *vis = (std::complex<double> *) vis_buf.ptr;

    for (int i=0; i < nu*nv; i++)
        vis[i] = 0;

    // Use only 1 thread first, otherwise Delaunator could have a segfault.

    #ifdef _OPENMP
    omp_set_num_threads(1);
    #endif

    // Set up the coordinates for the triangulation.

    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Now set to the appropriate number of threads for the remainder of the 
    // program.

    #ifdef _OPENMP
    omp_set_num_threads(nthreads);
    #endif

    // Run the kernel on the CPU.
    if (flux_buf.ndim == 2) {
        if (mode == "extended")
            ft2D_extended(x, y, flux, u, v, vis, nu, nv, d.triangles, nthreads);
        else
            ft2D(x, y, flux, u, v, vis, nu, nv, d.triangles, nthreads);
    }
    else {
        if (mode == "extended")
            ft_extended(x, y, flux, u, v, vis, nu, d.triangles, nthreads);
        else
            ft(x, y, flux, u, v, vis, nu, d.triangles, nthreads);
    }

    // Do the centering of the data.

    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 2> center(-dx, -dy);

    #pragma omp parallel for
    for (std::size_t i = 0; i < (std::size_t) nu; i++) {
        Vector <double, 2> uv(2*pi*u[i], 2*pi*v[i]);

        if (flux_buf.ndim == 2) {
            for (std::size_t j = 0; j < (std::size_t) nv; j++) {
                vis[i*nv+j] = vis[i*nv+j] * (cos(center.dot(uv)) +
                    I*sin(center.dot(uv)));
            }
        }
        else {
            vis[i] = vis[i] * (cos(center.dot(uv)) + I*sin(center.dot(uv)));
        }
    }

    return _vis;
}

PYBIND11_MODULE(cpu, m) {
    m.def("trift", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads, 
                std::string mode) {
            py::gil_scoped_release release;
            return trift(x, y, flux, u, v, dx, dy, nthreads, mode);},
            "Triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1,
            py::arg("mode")="basic");
}
