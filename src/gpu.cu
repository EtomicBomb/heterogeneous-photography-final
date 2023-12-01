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

// TODO: fix [d, r, c] -> [d * cols_src * rows + r * cols_src + c]
// index out of bounds, especially in patch_similarity
// thread, block index out of bounds of array
// pixel similarity based on something less aggressive than x^2
// launch kernels with the right grid / block shape
// parallelize cumulative sum functions? not sure if that's needed
// profile to see what bottlenecks are
// is float range / precision large enough to handle the fact that we are doing the cumulative sum for 2d array? 
// final result: 
//      a map of (rows, cols_src) -> positive ssize_t disparity
//      along with (rows, cols_src) -> occlusion map (0 fine, 1 left, 2 right)? or just (0, 1)?
__global__
void find_correspondances(ssize_t irows, ssize_t icols, ssize_t ocols, float *matrix, float *out) {
    size_t i = threadIdx.x;
    size_t j = blockIdx.x;
    for (ssize_t k = 0; k < irows + icols; k++) {
        if (i + j == k) {
            out[i*ocols + j] = min3(out[(i-1)*ocols + (j-1)], out[(i-1)*ocols + j], out[i + (j-1)]) + matrix[x + y*icols]
        }
    }
    ssize_t r = blockIdx.y;
    ssize_t c = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__
void pixel_similarity(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, float *src, float *dst, float *pixel_similarity) {
    ssize_t d = blockIdx.z * blockDim.z + threadIdx.z;
    ssize_t r = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t c = blockIdx.x * blockDim.x + threadIdx.x;
    float src_value = src[r, c];
    float dst_value = c + d < cols_dst ? dst[r, c + d] : 0.0;
    float distance = src_value - dst_value;
    pixel_similarity[d, r, c] = distance * distance;
}

__global__
void cumulative_sum_cols_src(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, float *array) {
    ssize_t d = blockIdx.z * blockDim.z + threadIdx.z;
    ssize_t r = blockIdx.y * blockDim.y + threadIdx.y;
    for (ssize_t c = 1; c < cols_src; c++) {
        array[d, r, c] += array[d, r, c - 1];
    }
}

__global__
void cumulative_sum_rows(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, float *array) {
    ssize_t d = blockIdx.z * blockDim.z + threadIdx.z;
    ssize_t c = blockIdx.x * blockDim.x + threadIdx.x;
    for (ssize_t r = 1; r < rows; r++) {
        array[d, r, c] += array[d, r - 1, c];
    }
}

__global__
void patch_similarity(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, ssize_t patch_size, float *pixel_similarity, float *patch_similarity) {
    ssize_t d = blockIdx.z * blockDim.z + threadIdx.z;
    ssize_t r = blockIdx.y * blockDim.y + threadIdx.y;
    ssize_t c = blockIdx.x * blockDim.x + threadIdx.x;
    ssize_t c0 = c - patch_size;
    ssize_t c1 = c + patch_size;
    ssize_t r0 = r - patch_size;
    ssize_t r1 = r + patch_size;
    patch_similarity[d, r, c] = 
        pixel_similarity[d, r1, c1] // index out of bounds
        - pixel_similarity[d, r0, c1]
        - pixel_similarity[d, r1, c0]
        + pixel_similarity[d, r0, c0];
}

__global__
void traceback_disparity(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, float *correspondance_cost, int *disparity, int *occlusion) {
    ssize_t r = blockIdx.y * blockDim.y + threadIdx.y;

    ssize_t d = 0;
    for (ssize_t c = 0; c < cols_src; c++) {
        min3();
        disparity[r, c] = d;
        int condition;
        occlusion[r, c] = condition;
        d = something;
    }
}

extern "C" double 
scanline_stereo(ssize_t cols_dst, ssize_t rows, ssize_t cols_src, ssize_t patch_size, float *src, float *dst, int *disparity, int *occlusion) {
    ssize_t ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    dim3 block_drc(32, 32, 1);
    dim3 grid_drc((cols + block_drc.x - 1) / block_drc.x, rows);
    // drc, dc, dr, r

    float *pixel_similarity, *patch_similarity, *correspondance_cost;
    cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(float));
    cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(float));
    cudaMalloc(&correspondance_cost, rows * cols_src * cols_dst * sizeof(float));

    int *disparity_device, *occlusion_device;
    cudaMalloc(&disparity_device, rows * cols_src * sizeof(int));
    cudaMalloc(&occlusion_device, rows * cols_src * sizeof(int));

    auto start = std::chrono::high_resolution_clock::now();

    pixel_similarity<<<grid_drc, block_drc, 0, 0>>>(cols_dst, rows, cols_src, src, dst, pixel_similarity);
    cumulative_sum_cols_src<<<grid_dr, block_dr, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    cumulative_sum_rows<<<grid_dc, block_dc, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    patch_similarity<<<grid_drc, block_drc, 0, 0>>>(cols_dst, rows, cols_src, patch_size, pixel_similarity, patch_similarity);
    find_correspondances<<<grid_shape, block_shape, 0, 0>>>(cols_dst, rows, cols_src, patch_similarity, correspondance_cost);
    traceback_disparity<<<grid_r, block_r, 0, 0>>>(cols_dst, rows, cols_src, correspondance_cost, disparity_device, occlusion_device);

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    cudaMemcpy(disparity, disparity_device, rows * cols_src * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(occlusion, occlusion_device, rows * cols_src * sizeof(int), cudaMemcpyDeviceToHost);

    return elapsed;
}



//////////////////////////// SAMPLE CODE //////////////////////////////////////////

__global__
void matrix_vector_kernel(ssize_t rows, ssize_t cols, float *matrix, float *vector, float *result) {
    extern __shared__ float shared_memory[];

    ssize_t r = blockIdx.y;
    ssize_t c = blockIdx.x * blockDim.x + threadIdx.x;

    float warp_sum = c >= cols ? 0.0 : matrix[r * cols + c] * vector[c];

    __syncwarp();
    for (ssize_t offset = warpSize/2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_memory[threadIdx.x / warpSize] = warp_sum;
    }

     __syncthreads();
    float block_sum = 0.0;
    for (ssize_t i = 0; i < blockDim.x / warpSize; i++) {
        block_sum += shared_memory[i];
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[r], block_sum);
    }
}

extern "C" float 
matrix_vector(ssize_t problem_count, ssize_t rows, ssize_t cols, float *matrix, float *vector, float *result) {
    ssize_t ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    float *matrix_d, *vector_d, *result_d;
    cudaMalloc(&matrix_d, problem_count * rows * cols * sizeof(float));
    cudaMalloc(&vector_d, problem_count * cols * sizeof(float));
    cudaMalloc(&result_d, problem_count * rows * sizeof(float));

    cudaMemcpy(matrix_d, matrix, problem_count * rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_d, vector, problem_count * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_shape(1024, 1);
    dim3 grid_shape((cols + block_shape.x - 1) / block_shape.x, rows);
    ssize_t shared_memory = 256 / 32 * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();
    for (ssize_t i = 0; i < problem_count; i++) {
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

    cudaMemcpy(result, result_d, problem_count * rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(matrix_d);
    cudaFree(vector_d);
    cudaFree(result_d);

    return elapsed / problem_count;
}

