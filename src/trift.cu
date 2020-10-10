#include "trift.h"
#include <delaunator-header-only.hpp>

__global__
void ft(double *x, double *y, double *flux, double *u, double *v, 
        thrust::complex<double> *vis, int nu, int* triangles, int nt) {
    // Loop through and take the Fourier transform of each triangle.
    
    thrust::complex<double> I = thrust::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < nu; k += stride) {
        Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

        // Loop over all of the triangles for this uv point.

        for (int i = 0; i < nt; i+=3) {
            double intensity_triangle = (flux[triangles[i]] + 
                flux[triangles[i+1]] + flux[triangles[i+2]]) / 3.;

            for (int j = 0; j < 3; j++) {
                // Calculate the vectors for the vertices of the triangle.

                int i_rn1 = triangles[i + (j+1)%3];
                Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

                int i_rn = triangles[i + j];
                Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

                int i_rn_1 = triangles[i + (j+2)%3];
                Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

                // Calculate the vectors for the edges of the triangle.

                Vector<double, 3> ln = rn1 - rn;
                Vector<double, 3> ln_1 = rn - rn_1;

                // Now loop through the UV points and calculate the Fourier
                // Transform.

                double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));
                double rn_dot_uv = rn.dot(uv);

                double sinVal, cosVal;
                sincos(rn_dot_uv, &sinVal, &cosVal);

                vis[k] += intensity_triangle * ln_1_dot_zhat_cross_ln /
                        (ln.dot(uv) * ln_1.dot(uv)) * (cosVal - I*sinVal);
            }
        }
    }
}

__global__
void ft_extended(double *x, double *y, double *flux, double *u, double *v, 
        thrust::complex<double> *vis, int nu, int* triangles, int nt) {
    // Loop through and take the Fourier transform of each triangle.
    
    thrust::complex<double> I = thrust::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < nu; k += stride) {
        Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

        // Loop through the triangles and calculate the Fourier transform.

        for (int i = 0; i < nt; i+=3) {
            // Calculate the area of the triangle.

            Vector<double, 3> Vertex1(x[triangles[i+0]], 
                    y[triangles[i+0]], 0.);
            Vector<double, 3> Vertex2(x[triangles[i+1]], 
                    y[triangles[i+1]], 0.);
            Vector<double, 3> Vertex3(x[triangles[i+2]], 
                    y[triangles[i+2]], 0.);

            Vector<double, 3> Side1 = Vertex2 - Vertex1;
            Vector<double, 3> Side2 = Vertex3 - Vertex1;

            double Area = 0.5 * (Side1.cross(Side2)).norm();

            // Precompute some aspects of the integral that remain the same.

            Vector<thrust::complex<double>, 3> integral_part1(0., 0., 0.);
            thrust::complex<double> integral_part2 = 0.;

            for (int m = 0; m < 3; m++) {
                // Get the appropriate vertices.

                int i_rm1 = triangles[i + (m+1)%3];
                Vector<double, 3> rm1(x[i_rm1], y[i_rm1], 0.);

                int i_rm = triangles[i + m];
                Vector<double, 3> rm(x[i_rm], y[i_rm], 0.);

                // Calculate the needed derivatives of those.

                Vector<double, 3> lm = rm1 - rm;
                Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
                Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

                // Now loop through the uv points and calculate the pieces of 
                // the integral.

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = j0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        j1(uv.dot(lm)/2.);

                double sinVal, cosVal;
                sincos(r_mc.dot(uv), &sinVal, &cosVal);

                thrust::complex<double> exp_part = (cosVal -  
                        I*sinVal) / (uv.dot(uv));

                // Now add everything together.

                integral_part1 += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2 += -I*bessel0_prefix3 * bessel0 * exp_part;
            }

            // Now loop through an do the actual calculation.

            for (int j = 0; j < 3; j++) {
                double intensity = flux[triangles[i + j]];

                // Calculate the vectors for the vertices of the triangle.

                int i_rn1 = triangles[i + (j+1)%3];
                Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

                int i_rn = triangles[i + j];
                Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

                int i_rn_1 = triangles[i + (j+2)%3];
                Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

                // Calculate the vectors for the edges of the triangle.

                Vector<double, 3> ln1 = rn_1 - rn1;

                Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

                // Finally put into the real and imaginary components.

                vis[k] += intensity * zhat_cross_ln1.dot(integral_part1+
                        rn1*integral_part2) / (2.*Area);
            }
        }
    }
}

__global__
void ft2D(double *x, double *y, double *flux, double *u, double *v, 
        thrust::complex<double> *vis, int nu, int nv, int* triangles, int nt) {
    // Loop through and take the Fourier transform of each triangle.
    
    thrust::complex<double> I = thrust::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < nu; k += stride) {
        Vector <double, 3> uv(2*pi*u[k], 2*pi*v[k], 0.);

        double *intensity_triangle = new double[nv];

        for (int i = 0; i < nt; i+=3) {
            // Get the intensity of the triangle at each wavelength.

            for (int l = 0; l < nv; l++) {
                intensity_triangle[l] = (flux[triangles[i]*nv+l] + 
                    flux[triangles[i+1]*nv+l] + flux[triangles[i+2]*nv+l]) / 3.;
            }

            // Calculate the FT

            for (int j = 0; j < 3; j++) {
                // Calculate the vectors for the vertices of the triangle.

                int i_rn1 = triangles[i + (j+1)%3];
                Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

                int i_rn = triangles[i + j];
                Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

                int i_rn_1 = triangles[i + (j+2)%3];
                Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

                // Calculate the vectors for the edges of the triangle.

                Vector<double, 3> ln = rn1 - rn;
                Vector<double, 3> ln_1 = rn - rn_1;

                // Now loop through the UV points and calculate the Fourier
                // Transform.

                double ln_1_dot_zhat_cross_ln = ln_1.dot(zhat.cross(ln));
                double rn_dot_uv = rn.dot(uv);
                
                double sinVal, cosVal;
                sincos(rn_dot_uv, &sinVal, &cosVal);

                int idy = k * nv;

                for (std::size_t l = 0; l < (std::size_t) nv; l++) {
                    vis[idy+l] += intensity_triangle[l] * 
                        ln_1_dot_zhat_cross_ln / (ln.dot(uv) * ln_1.dot(uv)) * 
                        (cosVal - I*sinVal);
                }
            }
        }

        // Clean up.

        delete[] intensity_triangle;
    }
}

__global__
void ft2D_extended(double *x, double *y, double *flux, double *u, double *v, 
        thrust::complex<double> *vis, int nu, int nv, int* triangles, int nt) {
    // Loop through and take the Fourier transform of each triangle.
    
    thrust::complex<double> I = thrust::complex<double>(0., 1.);
    Vector<double, 3> zhat(0., 0., 1.);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int k = index; k < nu; k += stride) {
        Vector <double, 3> uv(2*pi*u[k],2*pi*v[k],0.);

        // Loop through the triangles and calculate the Fourier transform.

        for (int i = 0; i < nt; i+=3) {
            // Calculate the area of the triangle.

            Vector<double, 3> Vertex1(x[triangles[i+0]], y[triangles[i+0]], 0.);
            Vector<double, 3> Vertex2(x[triangles[i+1]], y[triangles[i+1]], 0.);
            Vector<double, 3> Vertex3(x[triangles[i+2]], y[triangles[i+2]], 0.);

            Vector<double, 3> Side1 = Vertex2 - Vertex1;
            Vector<double, 3> Side2 = Vertex3 - Vertex1;

            double Area = 0.5 * (Side1.cross(Side2)).norm();

            // Precompute some aspects of the integral that remain the same.

            Vector<thrust::complex<double>, 3> integral_part1(0.,0.,0.);
            thrust::complex<double> integral_part2 = 0.;

            for (int m = 0; m < 3; m++) {
                // Get the appropriate vertices.

                int i_rm1 = triangles[i + (m+1)%3];
                Vector<double, 3> rm1(x[i_rm1], y[i_rm1],  0.);

                int i_rm = triangles[i + m];
                Vector<double, 3> rm(x[i_rm], y[i_rm],  0.);

                // Calculate the needed derivatives of those.

                Vector<double, 3> lm = rm1 - rm;
                Vector<double, 3> r_mc = 0.5 * (rm1 + rm);
                Vector<double, 3> zhat_cross_lm = zhat.cross(lm);

                // Now loop through the uv points and calculate the pieces of 
                // the integral.

                double zhat_dot_lm_cross_uv = zhat.dot(lm.cross(uv));

                // The various parts of the long integral equation, just
                // split up for ease of reading.

                Vector<double, 3> bessel0_prefix1 = zhat_cross_lm;
                Vector<double, 3> bessel0_prefix2 = r_mc * zhat_dot_lm_cross_uv;
                double bessel0_prefix3 = zhat_dot_lm_cross_uv;
                Vector<double, 3> bessel0_prefix4 = 2.*uv/uv.dot(uv)*
                        zhat_dot_lm_cross_uv;

                double bessel0 = j0(uv.dot(lm)/2.);

                Vector<double, 3> bessel1 = lm * (zhat_dot_lm_cross_uv/2.) * 
                        j1(uv.dot(lm)/2.);

                double sinVal, cosVal;
                sincos(r_mc.dot(uv), &sinVal, &cosVal);

                thrust::complex<double> exp_part = (cosVal -  
                        I*sinVal) / (uv.dot(uv));

                // Now add everything together.

                integral_part1 += ((bessel0_prefix1 + I*bessel0_prefix2 - 
                        bessel0_prefix4) * bessel0 - bessel1) * exp_part;
                integral_part2 += -I*bessel0_prefix3 * bessel0 * exp_part;
            }

            // Now loop through an do the actual calculation.

            for (int j = 0; j < 3; j++) {
                // Calculate the vectors for the vertices of the triangle.

                int i_rn1 = triangles[i + (j+1)%3];
                Vector<double, 3> rn1(x[i_rn1], y[i_rn1],  0.);

                int i_rn = triangles[i + j];
                Vector<double, 3> rn(x[i_rn], y[i_rn],  0.);

                int i_rn_1 = triangles[i + (j+2)%3];
                Vector<double, 3> rn_1(x[i_rn_1], y[i_rn_1],  0.);

                // Calculate the vectors for the edges of the triangle.

                Vector<double, 3> ln1 = rn_1 - rn1;

                Vector<double, 3> zhat_cross_ln1 = zhat.cross(ln1);

                // Finally put into the real and imaginary components.

                int idy = k * nv;

                for (int l = 0; l < (std::size_t) nv; l++) {
                    vis[idy+l] += flux[triangles[i+j]*nv+l] * 
                            zhat_cross_ln1.dot(integral_part1+rn1*
                            integral_part2) / (2.*Area);
                }
            }
        }
    }
}

py::array_t<std::complex<double>> trift(py::array_t<double> _x,
        py::array_t<double> _y, py::array_t<double> _flux,
        py::array_t<double> _u, py::array_t<double> _v, double dx, double dy,
        int nthreads, std::string mode) {

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

    auto _vis = py::array_t<std::complex<double>>(nu*nv);
    if (flux_buf.ndim == 2) _vis.resize({nu, nv});

    auto vis_buf = _vis.request();
    std::complex<double> *vis = (std::complex<double> *) vis_buf.ptr;

    for (int i=0; i < nu*nv; i++)
        vis[i] = 0;

    // Properly pass arrays to the GPU.
    double *d_x, *d_y, *d_flux, *d_u, *d_v;
    cudaMalloc(&d_x, nx*sizeof(double));
    cudaMalloc(&d_y, nx*sizeof(double));
    cudaMalloc(&d_flux, nx*sizeof(double));
    cudaMalloc(&d_u, nu*sizeof(double));
    cudaMalloc(&d_v, nu*sizeof(double));

    std::complex<double> *d_vis;
    cudaMalloc(&d_vis, nu*nv*sizeof(std::complex<double>));

    cudaMemcpy(d_x, x, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flux, flux, nx*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, nu*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, nu*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vis, vis, nu*nv*sizeof(std::complex<double>), 
            cudaMemcpyHostToDevice);

    // Set up the coordinates for the triangulation.

    std::vector<double> coords;

    for (int i=0; i < nx; i++) {
        coords.push_back(x[i]);
        coords.push_back(y[i]);
    }

    // Run the Delauney triangulation here.

    delaunator::Delaunator d(coords);

    // Properly set up the triangles on the GPU.

    int nt = d.triangles.size();
    int *triangles;
    cudaMallocManaged(&triangles, nt*sizeof(int));
    for (int i = 0; i < nt; i++)
        triangles[i] = (int) d.triangles[i];

    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(triangles, nt*sizeof(int), device, NULL);

    // Run the kernel on the GPU.
    int blockSize = 32;
    int numBlocks = (nu + blockSize - 1) / blockSize;

    if (flux_buf.ndim == 2) {
        if (mode == "extended")
            ft2D_extended<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, d_u, d_v, 
                    (thrust::complex<double> *) d_vis, nu, nv, triangles, nt);
        else
            ft2D<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, d_u, d_v, 
                    (thrust::complex<double> *) d_vis, nu, nv, triangles, nt);
    }
    else {
        if (mode == "extended")
            ft_extended<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, d_u, d_v, 
                    (thrust::complex<double> *) d_vis, nu, triangles, nt);
        else
            ft<<<numBlocks, blockSize>>>(d_x, d_y, d_flux, d_u, d_v, 
                    (thrust::complex<double> *) d_vis, nu, triangles, nt);
    }

    // Retrieve the updated values of vis from the GPU.
    cudaMemcpy(vis, d_vis, nu*nv*sizeof(std::complex<double>), 
            cudaMemcpyDeviceToHost);

    // Clean up the device arrays.
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_flux);
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_vis);
    cudaFree(triangles);

    // Do the centering of the data.

    std::complex<double> I = std::complex<double>(0., 1.);
    Vector<double, 2> center(-dx, -dy);

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

PYBIND11_MODULE(cuda, m) {
    m.def("trift", [](py::array_t<double> x, py::array_t<double> y, 
                py::array_t<double> flux, py::array_t<double> u, 
                py::array_t<double> v, double dx, double dy, int nthreads, 
                std::string mode) {
            return trift(x, y, flux, u, v, dx, dy, nthreads, mode);},
            "Triangulated Fourier Transform.", 
            py::arg("x"), py::arg("y"), py::arg("flux"), py::arg("u"), 
            py::arg("v"), py::arg("dx"), py::arg("dy"), py::arg("nthreads")=1, 
            py::arg("mode")="basic");
}
