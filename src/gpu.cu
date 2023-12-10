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

__device__ long 
argmin3(double x, double y, double z) {
    if (x < y) {
        return x < z ? 0 : 2;
    } 
    return y < z ? 1 : 2;
}

__device__ double 
min3(double x, double y, double z) {
    long index = argmin3(x, y, z);
    double numbers[] = {x, y, z};
    return numbers[index];
}

// [ ] thread block nonmultiple edge case
// [ ] pixel similarity based on something less aggressive than x^2
// [ ] launch kernels with the right grid / block shape
// [ ] profile to see what bottlenecks are
// [ ] is double range / precision large enough to handle the fact that we are doing the cumulative sum for 2d array? 
// only process some subset of the rows at once to limit our memory usage
// parallelize cumulative sum functions? not sure if that's needed

__global__ void
find_costs(long rows, long cols_src, long cols_dst, double occlusion_cost, const double *patch_similarity, double *cost) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    for (long k = 0; k < cols_src + cols_dst; k++) {
        long d = k - s;
        if (r < rows && s < cols_src && d < cols_dst && d >= 0) {
            if (s == 0 || d == 0) {
                cost I(r, s, d) = max(s, d) * occlusion_cost;
            } else {
                double match = cost I(r, s - 1, d - 1) + patch_similarity I(r, s, d);
                double left = cost I(r, s, d - 1) + occlusion_cost;
                double up = cost I(r, s - 1, d) + occlusion_cost;
                cost I(r, s, d) = min3(match, left, up);
            }
        }
        __syncthreads();
    }
}

__global__ void
traceback_correspondence(long rows, long cols_src, long cols_dst, const double *cost, long *correspondence, char *valid) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows) return;

    long s = cols_src - 1;
    long d = cols_dst - 1;
    while (s != 0 && d != 0) { // yes
        double match = cost I(r, s - 1, d - 1);
        double left = cost I(r, s, d - 1);
        double up = cost I(r, s - 1, d);
        long direction = argmin3(match, left, up);
        long us[] = {1, 0, 1}; 
        long ud[] = {1, 1, 0};
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

__global__ void
get_patch_similarity(long rows, long cols_src, long cols_dst, long patch_size, const double *pixel_similarity, double *patch_similarity) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || s >= cols_src || d >= cols_dst) return;

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

    patch_similarity I(r, s, d) = sum / count;
}

__global__ void
get_patch_similarity_naive(long rows, long cols_src, long cols_dst, long patch_size, const double *src, const double *dst, double *patch_similarity) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;
    long d = blockIdx.z * blockDim.z + threadIdx.z;

    if (r >= rows || s >= cols_src || d >= cols_dst) return;

    double sum = 0.0;
    double count = 0.0;
    for (long rn = r - patch_size; rn <= r + patch_size; rn++) {
        for (long sn = s - patch_size; sn <= s + patch_size; sn++) {
            long dn = sn + d - s;
            if (rn >= rows || rn < 0 || sn >= cols_src || sn < 0 || dn < 0 || dn >= cols_dst) continue;
            double distance = src Isrc(rn, sn) - dst Idst(rn, dn);
            sum += distance * distance;
            count += 1;
        }
    }

    patch_similarity I(r, s, d) = sum / count;
}

extern "C" int
scanline_stereo_naive(long rows, long cols_src, long cols_dst, long patch_size, double occlusion_cost, const double *src, const double *dst, long *correspondence, char *valid, float *timings) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return -1;
    }
    cudaSetDevice(0);

    double *src_device;
    cudaMalloc(&src_device, rows * cols_src * sizeof(*src_device));
    cudaMemcpy(src_device, src, rows * cols_src * sizeof(*src_device), cudaMemcpyHostToDevice);
    double *dst_device;
    cudaMalloc(&dst_device, rows * cols_dst * sizeof(*dst_device));
    cudaMemcpy(dst_device, dst, rows * cols_dst * sizeof(*dst_device), cudaMemcpyHostToDevice);

    double *patch_similarity;
    CHECK(cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(*patch_similarity)));
    double *cost;
    CHECK(cudaMalloc(&cost, rows * cols_src * cols_dst * sizeof(*cost)));
    long *correspondence_device; 
    CHECK(cudaMalloc(&correspondence_device, rows * cols_src * sizeof(*correspondence_device)));
    char *valid_device;
    CHECK(cudaMalloc(&valid_device, rows * cols_src * sizeof(*valid_device)));
    cudaMemset(&valid_device, 0, rows * cols_src * sizeof(*valid_device));

    dim3 block, grid;

    size_t timing_event_count = 4;
    std::vector<cudaEvent_t> events(timing_event_count);
    for (cudaEvent_t& event : events) {
        cudaEventCreate(&event);
    }
    cudaEventRecord(events[0]);

    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_patch_similarity_naive<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, patch_size, src_device, dst_device, patch_similarity);

    cudaEventRecord(events[1]);

    block = dim3(cols_src, 1, 1);
    grid = dim3(1, rows, 1);
    find_costs<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, occlusion_cost, patch_similarity, cost);

    cudaEventRecord(events[2]);

    block = dim3(1, 1024, 1);
    grid = dim3(1, (rows + block.y - 1) / block.y, 1);
    traceback_correspondence<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, cost, correspondence_device, valid_device);

    cudaEventRecord(events[3]);

    CHECK(cudaMemcpy(correspondence, correspondence_device, rows * cols_src * sizeof(*correspondence), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid, valid_device, rows * cols_src * sizeof(*valid), cudaMemcpyDeviceToHost));

    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(patch_similarity);
    cudaFree(cost);
    cudaFree(correspondence_device);
    cudaFree(valid_device);

    for (size_t i = 1; i < timing_event_count; i++) {
        cudaEventElapsedTime(&timings[i-1], events[0], events[i]);
    }

    return 0;

}

extern "C" int
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double occlusion_cost, const double *src, const double *dst, long *correspondence, char *valid, float *timings) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return -1;
    }
    cudaSetDevice(0);

    double *src_device;
    cudaMalloc(&src_device, rows * cols_src * sizeof(*src_device));
    cudaMemcpy(src_device, src, rows * cols_src * sizeof(*src_device), cudaMemcpyHostToDevice);
    double *dst_device;
    cudaMalloc(&dst_device, rows * cols_dst * sizeof(*dst_device));
    cudaMemcpy(dst_device, dst, rows * cols_dst * sizeof(*dst_device), cudaMemcpyHostToDevice);

    double *pixel_similarity, *patch_similarity, *cost;
    cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(*pixel_similarity));
    cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(*patch_similarity));
    cudaMalloc(&cost, rows * cols_src * cols_dst * sizeof(*cost));
    long *correspondence_device; 
    cudaMalloc(&correspondence_device, rows * cols_src * sizeof(*correspondence_device));
    char *valid_device;
    cudaMalloc(&valid_device, rows * cols_src * sizeof(*valid_device));
    cudaMemset(&valid_device, 0, rows * cols_src * sizeof(*valid_device));

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

    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_patch_similarity<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, patch_size, pixel_similarity, patch_similarity);

    cudaEventRecord(events[5]);

    block = dim3(cols_src, 1, 1);
    grid = dim3(1, rows, 1);
    find_costs<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, occlusion_cost, patch_similarity, cost);

    cudaEventRecord(events[6]);

    block = dim3(1, rows, 1);
    grid = dim3(1, (rows + block.y - 1) / block.y, 1);
    traceback_correspondence<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, cost, correspondence_device, valid_device);

    cudaEventRecord(events[7]);

    CHECK(cudaMemcpy(correspondence, correspondence_device, rows * cols_src * sizeof(*correspondence), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(valid, valid_device, rows * cols_src * sizeof(*valid), cudaMemcpyDeviceToHost));

    cudaFree(src_device);
    cudaFree(dst_device);
    cudaFree(pixel_similarity);
    cudaFree(patch_similarity);
    cudaFree(cost);
    cudaFree(correspondence_device);
    cudaFree(valid_device);

    for (size_t i = 1; i < timing_event_count; i++) {
        cudaEventElapsedTime(&timings[i-1], events[0], events[i]);
    }

    return 0;
}
