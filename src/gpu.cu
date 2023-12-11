#include <cuda.h>
#include <stdio.h>
#include <complex>
#include <cmath>
#include <stdio.h>
#include <sched.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>

#define I(r, s, d) [(r) * cols_dst * cols_src + (s) * cols_dst + (d)]
#define Isrc(r, s) [(r) * cols_src + (s)]
#define Idst(r, d) [(r) * cols_dst + (d)]

#define CHECK(e) do { cudaError_t err; if ((err = e) != cudaSuccess) { printf("line %d: %s\n", __LINE__, cudaGetErrorString(err)); exit(1);} } while (0)

__device__ inline char 
argmin3(double x, double y, double z) {
    if (x < y) {
        return x < z ? 0 : 2;
    } 
    return y < z ? 1 : 2;
}

__device__ inline void 
min3(double x, double y, double z, double *min, char *argmin) {
    double numbers[] = {x, y, z};
    char index = argmin3(x, y, z);
    *min = numbers[index];
    *argmin = index;
}

// [ ] thread block nonmultiple edge case
// [ ] pixel similarity based on something less aggressive than x^2
// [ ] launch kernels with the right grid / block shape
// [ ] profile to see what bottlenecks are
// [ ] is double range / precision large enough to handle the fact that we are doing the cumulative sum for 2d array? 
// only process some subset of the rows at once to limit our memory usage
// parallelize cumulative sum functions? not sure if that's needed

__device__ double
get_patch_similarity(long rows, long cols_src, long cols_dst, long patch_size, const double *pixel_similarity, long r, long s, long d) {
    long rm = r - patch_size - 1;
    long sm = s - patch_size - 1;
    long dm = d - patch_size - 1;

    long overflow = max(0l, max(s + patch_size - cols_src + 1, d + patch_size - cols_dst + 1));
    long underflow = - min(0l, min(sm, dm) + 1);

    long rp = min(r + patch_size, rows - 1);
    long sp = s + patch_size - overflow;
    long dp = d + patch_size - overflow;

    double rpsp = pixel_similarity I(rp, sp, dp);
    double rpsm = sm >= 0 && dm >= 0 ? pixel_similarity I(rp, sm, dm) : 0.0;
    double rmsp = rm >= 0 ? pixel_similarity I(rm, sp, dp): 0.0;
    double rmsm = rm >= 0 && sm >= 0 && dm >= 0 ? pixel_similarity I(rm, sm, dm) : 0.0;
    double sum = rpsp + rmsm - rpsm - rmsp;

    long width = rp - max(rm, -1l);
    long height = 2 * patch_size + 1 - underflow - overflow;
    long count = width * height;

    return sum / count;
}

__global__ void
find_costs(long rows, long cols_src, long cols_dst, long patch_size, double occlusion_cost, const double *pixel_similarity, double *cost, char *traceback, double *total) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    /*
    if (r == 0) {
        printf("s=%ld %d %d %d\n", s, blockIdx.x, blockDim.x, threadIdx.x);
    }
    */
    for (long k = 0; k < cols_src + cols_dst; k++) {
        long d = k - s;
        if (r < rows && s < cols_src && d < cols_dst && d >= 0) {
            if (s == 0 || d == 0) {
                cost I(r, s, d) = max(s, d) * occlusion_cost;
            } else {
                double patch_similarity = get_patch_similarity(rows, cols_src, cols_dst, patch_size, pixel_similarity, r, s, d);
                double match = cost I(r, s - 1, d - 1) + patch_similarity;
                double occlusion_src = cost I(r, s - 1, d) + occlusion_cost;
                double occlusion_dst = cost I(r, s, d - 1) + occlusion_cost;
                min3(match, occlusion_src, occlusion_dst, &cost I(r, s, d), &traceback I(r, s, d));
                if (r == 180) {
                    atomicAdd(total, traceback I (r, s, d));
                }
            }
        }
        __syncthreads();
    }
}

__global__ void
traceback_correspondence(long rows, long cols_src, long cols_dst, const double *cost, const char *traceback, long *correspondence, char *valid) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows) return;

    long s = cols_src - 1;
    long d = cols_dst - 1;
    while (s != 0 && d != 0) { // yes
        double match = cost I(r, s - 1, d - 1);
        double occlusion_src = cost I(r, s - 1, d);
        double occlusion_dst = cost I(r, s, d - 1);
        long direction = argmin3(match, occlusion_src, occlusion_dst);
        direction = traceback I(r, s, d);
        long us[] = {1, 1, 0}; 
        long ud[] = {1, 0, 1};
        s -= us[direction]; 
        d -= ud[direction]; 
        correspondence Isrc(r, s) = d;
        if (direction == 0) {
            valid Isrc(r, s) = 1;
        }
    }
}

__global__ void 
get_pixel_similarity(long rows, long cols_src, long cols_dst, const double *src, const double *dst, double *pixel_similarity) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || s >= cols_src || d >= cols_dst) return;

    double distance = src Isrc(r, s) - dst Idst(r, d);
    pixel_similarity I(r, s, d) = distance * distance;
}

__device__ void 
sum_diagonal(long rows, long cols_src, long cols_dst, long r, long s, long d, double *array) {
    s += 1;
    d += 1;
    while (s < cols_src && d < cols_dst) {
        array I(r, s, d) += array I(r, s - 1, d - 1);
        s += 1;
        d += 1;
    }
}

__global__ void 
cumulative_sum_cols_src(long rows, long cols_src, long cols_dst, double *array) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = 0;
    if (r >= rows || s >= cols_src) return;
    if (s == 0) return; // only want to sum central diagonal once
    sum_diagonal(rows, cols_src, cols_dst, r, s, d, array);
}

__global__ void 
cumulative_sum_cols_dst(long rows, long cols_src, long cols_dst, double *array) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = 0;
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    if (r >= rows || d >= cols_dst) return;
    sum_diagonal(rows, cols_src, cols_dst, r, s, d, array);
}

__global__ void 
cumulative_sum_rows(long rows, long cols_src, long cols_dst, double *array) {
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = blockIdx.z * blockDim.z + threadIdx.z;
    if (s >= cols_src || d >= cols_dst) return;
    for (long r = 1; r < rows; r++) {
        array I(r, s, d) += array I(r - 1, s, d);
    }
}

extern "C" int
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double occlusion_cost, const double *src, const double *dst, long *correspondence, char *valid, float *timings) {
    int ncuda_devices = 0;
    CHECK(cudaGetDeviceCount(&ncuda_devices));
    if (ncuda_devices == 0) {
        return -1;
    }
    cudaSetDevice(0);

    double *src_device;
    CHECK(cudaMalloc(&src_device, rows * cols_src * sizeof(*src_device)));
    CHECK(cudaMemcpy(src_device, src, rows * cols_src * sizeof(*src_device), cudaMemcpyHostToDevice));
    double *dst_device;
    CHECK(cudaMalloc(&dst_device, rows * cols_dst * sizeof(*dst_device)));
    CHECK(cudaMemcpy(dst_device, dst, rows * cols_dst * sizeof(*dst_device), cudaMemcpyHostToDevice));

    double *pixel_similarity;
    CHECK(cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(*pixel_similarity)));
    double *cost;
    CHECK(cudaMalloc(&cost, rows * cols_src * cols_dst * sizeof(*cost)));
    char *traceback;
    CHECK(cudaMalloc(&traceback, rows * cols_src * cols_dst * sizeof(*traceback)));
    long *correspondence_device; 
    CHECK(cudaMalloc(&correspondence_device, rows * cols_src * sizeof(*correspondence_device)));
    char *valid_device;
    CHECK(cudaMalloc(&valid_device, rows * cols_src * sizeof(*valid_device)));
    CHECK(cudaMemset(valid_device, 0, rows * cols_src * sizeof(*valid_device)));

    dim3 block, grid;

    size_t timing_event_count = 8;
    std::vector<cudaEvent_t> events(timing_event_count);
    for (cudaEvent_t& event : events) {
        cudaEventCreate(&event);
    }
    cudaEventRecord(events[0]);

    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_pixel_similarity<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, src_device, dst_device, pixel_similarity);

    cudaEventRecord(events[1]);

    block = dim3(1, 32, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, 1, (cols_dst + block.z - 1) / block.z);
    cumulative_sum_rows<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    cudaEventRecord(events[2]);

    block = dim3(32, 32, 1);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);
    cumulative_sum_cols_src<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);
    
    cudaEventRecord(events[3]);

    block = dim3(1, 32, 32);
    grid = dim3(1, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    cumulative_sum_cols_dst<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    cudaEventRecord(events[4]);
    
    CHECK(cudaMemset(traceback, 0, rows * cols_src * sizeof(*traceback)));
    double *iteration_count_d;
    CHECK(cudaMalloc(&iteration_count_d, sizeof(double)));
    CHECK(cudaMemset(iteration_count_d, 0, sizeof(double)));
    block = dim3(cols_src, 1, 1);
    grid = dim3(1, rows, 1);
    find_costs<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, patch_size, occlusion_cost, pixel_similarity, cost, traceback, iteration_count_d);
    std::vector<char> host_costs(rows * cols_src * cols_dst);
    CHECK(cudaMemcpy(host_costs.data(), traceback, rows * cols_src * cols_dst * sizeof(*host_costs.data()), cudaMemcpyDeviceToHost));
    double total_costs = 0;
    for (long r = 0; r < rows; r++) {
        for (long s = 0; s < cols_src; s++) {
            for (long d = 0; d < cols_dst; d++) {
                total_costs += host_costs.data() I(r, s, d);
            }
        }
    }
    double iteration_count;
    CHECK(cudaMemcpy(&iteration_count, iteration_count_d, sizeof(double), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(iteration_count_d));
    printf("total cost gpu: %.17g %.17g\n", total_costs, iteration_count);

    cudaEventRecord(events[6]);

    block = dim3(1, rows, 1);
    grid = dim3(1, (rows + block.y - 1) / block.y, 1);
    traceback_correspondence<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, cost, traceback, correspondence_device, valid_device);

    cudaEventRecord(events[7]);

    CHECK(cudaMemcpy(correspondence, correspondence_device, rows * cols_src * sizeof(*correspondence), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid, valid_device, rows * cols_src * sizeof(*valid), cudaMemcpyDeviceToHost));

    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(pixel_similarity);
    cudaFree(traceback);
    cudaFree(cost);
    cudaFree(correspondence_device);
    cudaFree(valid_device);

    for (size_t i = 1; i < timing_event_count; i++) {
        cudaEventElapsedTime(&timings[i-1], events[0], events[i]);
    }

    return 0;
}
