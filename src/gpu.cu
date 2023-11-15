#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <complex>
#include <cmath>
#include <stdio.h>
#include <sched.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>

__global__
void matrix_vector_kernel(int rows, int cols, double *matrix, double *vector, double *result) {
    extern __shared__ double shared_memory[];

    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    double warp_sum = c >= cols ? 0.0 : matrix[r * cols + c] * vector[c];

    __syncwarp();
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_memory[threadIdx.x / warpSize] = warp_sum;
    }

     __syncthreads();
    double block_sum = 0.0;
    for (int i = 0; i < blockDim.x / warpSize; i++) {
        block_sum += shared_memory[i];
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[r], block_sum);
    }
}

extern "C" double 
matrix_vector(int problem_count, int rows, int cols, double *matrix, double *vector, double *result) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    double *matrix_d, *vector_d, *result_d;
    cudaMalloc(&matrix_d, problem_count * rows * cols * sizeof(double));
    cudaMalloc(&vector_d, problem_count * cols * sizeof(double));
    cudaMalloc(&result_d, problem_count * rows * sizeof(double));

    cudaMemcpy(matrix_d, matrix, problem_count * rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_d, vector, problem_count * cols * sizeof(double), cudaMemcpyHostToDevice);

    dim3 block_shape(1024, 1);
    dim3 grid_shape((cols + block_shape.x - 1) / block_shape.x, rows);
    int shared_memory = 256 / 32 * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < problem_count; i++) {
        matrix_vector_kernel<<<grid_shape, block_shape, shared_memory, 0>>>(
                rows, 
                cols, 
                matrix_d + i * rows * cols, 
                vector_d + i * cols, 
                result_d + i * rows
        );
    }
    cudaDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    cudaMemcpy(result, result_d, problem_count * rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(matrix_d);
    cudaFree(vector_d);
    cudaFree(result_d);

    return elapsed / problem_count;
}

