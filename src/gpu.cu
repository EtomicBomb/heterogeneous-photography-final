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

#define I(r, s, d) [(r) * cols_dst * cols_src + (s) * cols_dst + (d)]
#define Irs(r, s) [(r) * cols_src + (s)]

__device__ long argmin3(double x, double y, double z) {
    if (x < y) {
        return x < z ? 0 : 2;
    } 
    return y < z ? 1 : 2;
}

__device__ long min3(double x, double y, double z) {
    long index = argmin3(x, y, z);
    double numbers[] = {x, y, z};
    return numbers[index];
}

__device__ long dmin(long x, long y) {
    return x < y ? x : y;
}

__device__ long dmax(long x, long y) {
    return x < y ? x : y;
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
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || s >= cols_src || d >= cols_dst) return;

    for (long k = 0; k < cols_src + cols_dst; k++) {
        if (s + d == k) {
            out I(r, s, d) = matrix I(r, s, d) +
                min3(out I(r, s-1, d-1), out I(r, s-1, d), out I(r, s, d-1));
        }
        __syncthreads();
    }
}

__global__ void
traceback_correspondance(long rows, long cols_src, long cols_dst, double *correspondance_cost, long *correspondance, char *occlusion) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows) return;

    long s = cols_src - 1;
    long d = cols_dst - 1;
    while (s != 0 || d != 0) {
        double up_left = correspondance_cost I(r, s - 1, d - 1);
        double left = correspondance_cost I(r, s, d - 1);
        double up = correspondance_cost I(r, s - 1, d);
        long direction = argmin3(up_left, up, left);
        long us[] = {-1, 0, -1}; 
        long ud[] = {-1, -1, 0};
        s += us[direction]; 
        d += ud[direction]; 
        correspondance Irs(r, s) = d;
        occlusion Irs(r, s) = direction != 0;
    }
}

__global__ void 
get_pixel_similarity(long rows, long cols_src, long cols_dst, double *src, double *dst, double *pixel_similarity) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= rows || s >= cols_src || d >= cols_dst) return;

    double src_value = src Irs(r, s);

    double dst_value = s + d < cols_dst ? dst Irs(r, s + d) : 0.0;
    double distance = src_value - dst_value;
    pixel_similarity I(r, s, d) = distance * distance;
}

__global__ void 
cumulative_sum_cols(long rows, long cols_src, long cols_dst, double *array) {
    // should range 0, ..., (cols_src + cols_dst)
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows || d >= cols_dst) return;

    long c_src_start = 0, c_dst_start = 0;
    if (d >= cols_src) {
        c_dst_start = d - cols_src;
    } else {
        c_src_start = d;
    }
    long bound = dmin(cols_src - c_src_start, cols_dst - c_dst_start);
    for (long c = 1; c < bound; c++) {
        array I(r, c_src_start + c, c_dst_start + c) += array I(r, c_src_start + c - 1, c_dst_start + c - 1);
    }
}

__global__ void 
cumulative_sum_rows(long rows, long cols_src, long cols_dst, double *array) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= cols_src || d >= cols_dst) return;
    for (long r = 1; r < rows; r++) {
        array I(r, s, d) += array I(r - 1, s, d);
    }
}

__global__ void 
get_patch_similarity(long rows, long cols_src, long cols_dst, long patch_size, double *pixel_similarity, double *patch_similarity) {
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    long rn = dmax(r - patch_size - 1, 0l);
    long sn = dmax(s - patch_size - 1, 0l);
    long dn = dmax(d - patch_size - 1, 0l);

    long rp = dmin(r + patch_size, rows);
    long sp = dmin(s + patch_size, cols_src);
    long dp = dmin(d + patch_size, cols_dst);

    double sum = pixel_similarity I(rn,sn,dn) 
        + pixel_similarity I(rp,sp,dp) 
        - pixel_similarity I(rp,sn,dn) 
        - pixel_similarity I(rn,sp,dp);
    double count = (rp - rn + 1) * dmin(sp - sn + 1, dp - dn + 1);
    patch_similarity I(r, s, d) = sum / count;
}

extern "C" double
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double *src, double *dst, long *correspondance, char *occlusion) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    dim3 block_rsd(1, 32, 32);
    dim3 grid_rsd(
            (rows + block_rsd.y - 1) / block_rsd.y, 
            (cols_src + block_rsd.x - 1) / block_rsd.x, 
            (cols_dst + block_rsd.z - 1) / block_rsd.z);
    dim3 block_rd(32, 1, 32);
    dim3 grid_rd(
            (rows + block_rd.y - 1) / block_rd.y, 
            1, 
            (cols_dst + block_rd.z - 1) / block_rd.z);
    dim3 block_sd(1, 32, 32);
    dim3 grid_sd(
            1, 
            (cols_src + block_sd.x - 1) / block_sd.x, 
            (cols_dst + block_sd.z - 1) / block_sd.z);
    dim3 block_r(1024, 1, 1);
    dim3 grid_r(
            (rows + block_r.y - 1) / block_r.y, 
            1, 
            1);

    double *pixel_similarity, *patch_similarity, *correspondance_cost;
    cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&correspondance_cost, rows * cols_src * cols_dst * sizeof(double));
    long *correspondance_device; 
    cudaMalloc(&correspondance_device, rows * cols_src * sizeof(long));
    char *occlusion_device;
    cudaMalloc(&occlusion_device, rows * cols_src * sizeof(char));

    auto start = std::chrono::high_resolution_clock::now();

    get_pixel_similarity<<<grid_rsd, block_rsd, 0, 0>>>(cols_dst, rows, cols_src, src, dst, pixel_similarity);
    cumulative_sum_cols<<<grid_rd, block_rd, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    cumulative_sum_rows<<<grid_sd, block_sd, 0, 0>>>(cols_dst, rows, cols_src, pixel_similarity);
    get_patch_similarity<<<grid_rsd, block_rsd, 0, 0>>>(cols_dst, rows, cols_src, patch_size, pixel_similarity, patch_similarity);
    find_correspondances<<<grid_rsd, block_rsd, 0, 0>>>(cols_dst, rows, cols_src, patch_similarity, correspondance_cost);
    traceback_correspondance<<<grid_r, block_r, 0, 0>>>(cols_dst, rows, cols_src, correspondance_cost, correspondance_device, occlusion_device);

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    cudaMemcpy(correspondance, correspondance_device, rows * cols_src * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(occlusion, occlusion_device, rows * cols_src * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(pixel_similarity);
    cudaFree(patch_similarity);
    cudaFree(correspondance_device);
    cudaFree(occlusion_device);

    return elapsed;
}



//////////////////////////// SAMPLE CODE //////////////////////////////////////////

__global__ void 
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

extern "C" double 
matrix_vector(long problem_count, long rows, long cols, double *matrix, double *vector, double *result) {
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

