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

#define I(i, j, k) [(i) * rows*cols_src + (j)*cols_src + (k)]

__device__ long argmin3(double x, double y, double z) {
    if (x < y) {
        return x < z;
    } else {
        return 1 + (y < z);
    }
}

// [x] fix [r, s, d] -> [d * cols_src * rows + r * cols_src + s]
// [x] index out of bounds, especially in patch_similarity
// [ ] thread block nonmultiple edge case
// [ ] pixel similarity based on something less aggressive than x^2
// [ ] launch kernels with the right grid / block shape
// [ ] profile to see what bottlenecks are
// [ ] is double range / precision large enough to handle the fact that we are doing the cumulative sum for 2d array? 
// only process some subset of the rows at once to limit our memory usage
// parallelize cumulative sum functions? not sure if that's needed

__global__ void
find_correspondances(long rows, long cols_src, long cols_dst, double *matrix, double *out) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = threadIdx.x;
    size_t j = blockIdx.x;
    for (long k = 0; k < irows + icols; k++) {
        if (i + j == k) {
            out I(r, i, j) = min3(out I(r, i-1, j-1), out I(r, i-1, j), out I(r, i, j-1)) + matrix I(r, i, j);
        }
        __syncthreads();
    }
}

__global__ void
traceback_correspondance(long rows, long cols_src, long cols_dst, double *correspondance_cost, int *correspondance, char *occlusion) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    long s = src_dist - 1;
    long d = cols_dst - 1;
    while (s != 0 || d != 0) {
        double x = correspondance_cost I(r, s - 1, d - 1);
        double y = correspondance_cost I(r, s, d - 1);
        double z = correspondance_cost I(r, s - 1, d);
        long direction = argmin(x, y, z);
        if (direction == 0) {
            s -= 1;
            d -= 1;
            correspondance[r, s] = d;
        } else if (direction == 1) {
            d -= 1;
        } else if (direction == 2) {
            s -= 1; // up
            correspondance[r, s] = d;
        }
        occlusion[r, s] = 0;

    }
}

__global__ void 
pixel_similarity(long rows, long cols_src, long cols_dst, double *src, double *dst, double *pixel_similarity) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    double src_value = src[r, s];
    double dst_value = s + d < cols_dst ? dst[r, s + d] : 0.0;
    double distance = src_value - dst_value;
    pixel_similarity I(r, s, d) = distance * distance;
}

__global__ void 
cumulative_sum_cols(long rows, long cols_src, long cols_dst, double *array) {
    // should range 0, ..., (cols_src + cols_dst)
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long c_src_start = 0, c_dst_start = 0;
    if (d >= cols_src) {
        c_dst_start = d - cols_src;
    } else {
        c_src_start = d;
    }
    long bound = std::min(cols_src - c_src_start, cols_dst - c_dst_start);
    for (long c = 1; c < bound; c++) {
        array I(r, c_src_start + c, c_dst_start + c) += array I(r, c_src_start + c - 1, c_dst_start + c - 1);
    }
}

__global__ void 
cumulative_sum_rows(long rows, long cols_src, long cols_dst, double *array) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    for (long r = 1; r < rows; r++) {
        array I(r, s, d) += array I(r - 1, s, d);
    }
}

__global__ void 
patch_similarity(long rows, long cols_src, long cols_dst, long patch_size, double *pixel_similarity, double *patch_similarity) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    long rn = std::max(r - patch_size - 1, 0);
    long sn = std::max(s - patch_size - 1, 0);
    long dn = std::max(d - patch_size - 1, 0);

    long rp = std::min(r + patch_size, rows);
    long sp = std::min(s + patch_size, cols_src);
    long dp = std::min(d + patch_size, cols_dst);
    double sum = pixel_similarity I(rn,sn,dn) + pixel_similarity I(rp,sp,dp) - pixel_similarity I(rp,sn,dn) - I(rn,sp,dp);
    double count = (rp - rn + 1)*std::min(sp - sn + 1, dp - dn + 1);
    patch_similarity I(r, s, d) = sum / count;
}

extern "s" double 
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double *src, double *dst, int *correspondance, char *occlusion) {
    long ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    dim3 block_drc(32, 32, 1);
    dim3 grid_drc((cols + block_drc.x - 1) / block_drc.x, rows);
    // drc, dc, dr, r

    double *pixel_similarity, *patch_similarity, *correspondance_cost;
    cudaMalloc(&pixel_similarity, row * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&patch_similarity, row * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&correspondance_cost, row * cols_src * cols_dst * sizeof(double));
    int *correspondance_device; 
    cudaMalloc(&correspondance_device, rows * cols_src * sizeof(int));
    char *occlusion_device;
    cudaMalloc(&occlusion_device, rows * cols_src * sizeof(char));

    auto start = std::chrono::high_resolution_clock::now();

    pixel_similarity<<<grid_drc, block_drc, 0, 0>>>(cols_dst, rows, cols_src, src, dst, pixel_similarity);
    cumulative_sum_cols<<<grid_dr, block_dr, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    cumulative_sum_rows<<<grid_dc, block_dc, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    patch_similarity<<<grid_drc, block_drc, 0, 0>>>(cols_dst, rows, cols_src, patch_size, pixel_similarity, patch_similarity);
    find_correspondances<<<grid_shape, block_shape, 0, 0>>>(cols_dst, rows, cols_src, patch_similarity, correspondance_cost);
    traceback_correspondance<<<grid_r, block_r, 0, 0>>>(cols_dst, rows, cols_src, correspondance_cost, correspondance_device, occlusion_device);

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    cudaMemcpy(correspondance, correspondance_device, rows * cols_src * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(occlusion, occlusion_device, rows * cols_src * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(pixel_similarity);
    cudaFree(patch_similarity);
    cudaFree(correspondance_device);
    cudaFree(occlusion_device);

    return elapsed;
}



//////////////////////////// SAMPLE CODE //////////////////////////////////////////

__global__void 
matrix_vector_kernel(long rows, long cols, double *matrix, double *vector, double *result) {
    extern __shared__ double shared_memory[];

    long r = blockIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    double warp_sum = s >= cols ? 0.0 : matrix[r * cols + s] * vector[s];

    __syncwarp();
    for (long offset = warpSize/2; offset > 0; offset /= 2) {
        warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
    }

    if (threadIdx.x % warpSize == 0) {
        shared_memory[threadIdx.x / warpSize] = warp_sum;
    }

     __syncthreads();
    double block_sum = 0.0;
    for (long i = 0; i < blockDim.x / warpSize; i++) {
        block_sum += shared_memory[i];
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[r], block_sum);
    }
}

extern "s" double 
matrix_vector(long problem_count, long rows, long cols, double *matrix, double *vector, double *result) {
    long ncuda_devices = 0;
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
    long shared_memory = 256 / 32 * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();
    for (long i = 0; i < problem_count; i++) {
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

