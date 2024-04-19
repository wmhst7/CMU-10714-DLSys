#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matrix_multiple(const float *X, const float *Y, float *Z, 
                        size_t m, size_t n, size_t p)
{
    /**
     * Dot product of matrix. X @ Y = Z
     * (m, p) @ (p, n) = (m, n)
     * Z[i, j] = Sum{X[i, k] * Y[k, j]} for k in [0, p)
     */

    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            for(size_t k = 0; k < p; k++){
                Z[i * n + j] += X[i * p + k] * Y[k * n + j];
            }
        }
    }
}

void softmax(float* X, size_t m, size_t n)
{
    /**
     * In-place softmax for axis 1.
     */
    for(size_t i = 0; i < m * n; i++) X[i] = exp(X[i]);
    float sum;
    for(size_t i = 0; i < m; i++){
        sum = 0.0;
        for(size_t j = 0; j < n; j++){
            sum += X[i * n + j];
        }
        for(size_t j = 0; j < n; j++){
            X[i * n + j] /= sum;
        }
    }
}


void transpose(float* X_T, const float* X, size_t m, size_t n)
{
    for(size_t i = 0; i < m; i++){
        for(size_t j = 0; j < n; j++){
            X_T[j * m + i] = X[i * n + j];
        }
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // X: (m, n)
    // X_bacth: (batch, n)
    // Z_batch: (batch, k)
    // Iy: (batch, k)
    // D: (n, k) 
    // theta: (n, k)
    float* Z_batch = new float[batch * n];
    float* X_T = new float[batch * n];
    float* D = new float[n * k];

    for(size_t start = 0; start < m; start += batch){
        size_t end = std::min(m, start + batch);
        size_t b = end - start;

        // Initialize Buffer
        memset(Z_batch, 0, sizeof(float) * batch * n);
        memset(X_T, 0, sizeof(float) * batch * n);
        memset(D, 0, sizeof(float) * k * n);
        const float *X_batch = &X[start * n];
        const unsigned char *y_batch = &y[start];

        // Z_batch = X_batch @ theta
        matrix_multiple(X_batch, theta, Z_batch, b, k, n);
        // Z_batch = softmax(Z_batch)
        softmax(Z_batch, b, k);
        // Z_batch -= Iy
        for(size_t i = 0; i < b; i++){
            Z_batch[i * k + y_batch[i]] -= 1;
        }
        // X_batch.T @ Z_batch
        transpose(X_T, X_batch, b, n);
        matrix_multiple(X_T, Z_batch, D, n, k, b);
        // theta -= lr * D
        for(size_t i = 0; i < n * k; i++){
            theta[i] -= lr * D[i] / b;
        }
    }

    delete[] Z_batch;
    delete[] X_T;
    delete[] D;
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
