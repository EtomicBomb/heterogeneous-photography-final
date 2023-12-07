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

__device__ long 
min3(double x, double y, double z) {
    long index = argmin3(x, y, z);
    double numbers[] = {x, y, z};
    return numbers[index];
}

__device__ long 
dmin(long x, long y) {
    return x < y ? x : y;
}

__device__ long 
dmax(long x, long y) {
    return x > y ? x : y;
}

// [ ] thread block nonmultiple edge case
// [ ] pixel similarity based on something less aggressive than x^2
// [ ] launch kernels with the right grid / block shape
// [ ] profile to see what bottlenecks are
// [ ] is double range / precision large enough to handle the fact that we are doing the cumulative sum for 2d array? 
// only process some subset of the rows at once to limit our memory usage
// parallelize cumulative sum functions? not sure if that's needed

__global__ void
find_correspondances(long rows, long cols_src, long cols_dst, const double *patch_similarity, double *correspondance_cost) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    for (long k = 0; k < cols_src + cols_dst; k++) {
        long d = k - s;
        if (r < rows && s < cols_src && d < cols_dst && d >= 0) {
            double up_left = s - 1 >= 0 && d - 1 >= 0 ? correspondance_cost I(r, s - 1, d - 1) : INFINITY;
            double left = d - 1 >= 0 ? correspondance_cost I(r, s, d - 1) : INFINITY;
            double up = s - 1 >= 0 ? correspondance_cost I(r, s - 1, d) : INFINITY;
            correspondance_cost I(r, s, d) = patch_similarity I(r, s, d) + min3(up_left, left, up); // this doesn't make sense because top left is gonna get infinity
        }
        __syncthreads();
    }
}

__global__ void
traceback_correspondance(long rows, long cols_src, long cols_dst, const double *correspondance_cost, long *correspondance, char *valid) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= rows) return;

    long s = cols_src - 1;
    long d = cols_dst - 1;
    while (s != 0 || d != 0) {
        double up_left = s - 1 >= 0 && d - 1 >= 0 ? correspondance_cost I(r, s - 1, d - 1) : INFINITY;
        double left = d - 1 >= 0 ? correspondance_cost I(r, s, d - 1) : INFINITY;
        double up = s - 1 >= 0 ? correspondance_cost I(r, s - 1, d) : INFINITY;
        long direction = argmin3(up_left, left, up);
        long us[] = {1, 0, 1}; 
        long ud[] = {1, 1, 0};
        s -= us[direction]; 
        d -= ud[direction]; 
        correspondance Isrc(r, s) = d;
        valid Isrc(r, s) = direction == 0;
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

    long rp = dmin(r + patch_size, rows - 1);
	long overflow = dmax(0, dmax(s + patch_size - (cols_src - 1), d + patch_size - (cols_dst - 1)));
    long sp = s + patch_size - overflow;
    long dp = d + patch_size - overflow;

	double rpspdp = pixel_similarity I(rp, sp, dp);
	double rpsmdm = sm >= 0 && dm >= 0 ? pixel_similarity I(rp, sm, dm) : 0.0;
	double rmspdp = rm >= 0 ? pixel_similarity I(rm, sp, dp): 0.0;
	double rmsmdm = rm >= 0 && sm >= 0 && dm >= 0 ? pixel_similarity I(rm, sm, dm) : 0.0;

    double sum = rpspdp + rmsmdm - rpsmdm - rmspdp;

	long underflow = - dmin(0, dmin(sm, dm) + 1);
	long overflow_count = dmax(0, dmax(s + patch_size - (cols_src - 1), d + patch_size - (cols_dst - 1)));
    long count = (rp - dmax(rm, -1)) * (2 * patch_size + 1 - underflow - overflow_count);
    patch_similarity I(r, s, d) = sum / count;
}

__global__ void
get_patch_similarity2(long rows, long cols_src, long cols_dst, long patch_size, const double *src, const double *dst, double *patch_similarity) {
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

extern "C" double
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, const double *src, const double *dst, long *correspondance, char *valid, double *result) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    double *src_device;
    cudaMalloc(&src_device, rows * cols_src * sizeof(*src_device));
    cudaMemcpy(src_device, src, rows * cols_src * sizeof(*src_device), cudaMemcpyHostToDevice);
    double *dst_device;
    cudaMalloc(&dst_device, rows * cols_dst * sizeof(*dst_device));
    cudaMemcpy(dst_device, dst, rows * cols_dst * sizeof(*dst_device), cudaMemcpyHostToDevice);

    double *pixel_similarity, *patch_similarity, *correspondance_cost;
    cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(*pixel_similarity));
    cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(*patch_similarity));
    cudaMalloc(&correspondance_cost, rows * cols_src * cols_dst * sizeof(*correspondance_cost));
    long *correspondance_device; 
    cudaMalloc(&correspondance_device, rows * cols_src * sizeof(*correspondance_device));
    cudaMemset(&correspondance_device, 0, rows * cols_src * sizeof(*correspondance_device));
    char *valid_device;
    cudaMalloc(&valid_device, rows * cols_src * sizeof(*valid_device));
    cudaMemset(&valid_device, 0, rows * cols_src * sizeof(*valid_device));

    auto start = std::chrono::high_resolution_clock::now(); 
    dim3 block, grid;

    /*
    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_pixel_similarity<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, src_device, dst_device, pixel_similarity);

    block = dim3(1, 32, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, 1, (cols_dst + block.z - 1) / block.z);
    cumulative_sum_rows<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    block = dim3(32, 32, 1);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, 1);
    cumulative_sum_cols_src<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    block = dim3(1, 32, 32);
    grid = dim3(1, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    cumulative_sum_cols_dst<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_patch_similarity<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, patch_size, pixel_similarity, patch_similarity);
    */

	double *patch22;
    cudaMalloc(&patch22, rows * cols_src * cols_dst * sizeof(*patch_similarity));
    block = dim3(32, 1, 32);
    grid = dim3((cols_src + block.x - 1) / block.x, (rows + block.y - 1) / block.y, (cols_dst + block.z - 1) / block.z);
    get_patch_similarity2<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, patch_size, src_device, dst_device, patch22);
	double *result2 = (double *)malloc(rows * cols_src * cols_dst * sizeof(*patch_similarity));
    CHECK(cudaMemcpy(result2, patch22, rows * cols_src * cols_dst * sizeof(*result), cudaMemcpyDeviceToHost));
    
    /* block = dim3(cols_src, 1, 1); */
    /* grid = dim3(1, rows, 1); */
    /* find_correspondances<<<block, grid, 0, 0>>>(rows, cols_src, cols_dst, patch_similarity, correspondance_cost); */

    /* block = dim3(1, 1024, 1); */
    /* grid = dim3(1, (rows + block.y - 1) / block.y, 1); */
    /* traceback_correspondance<<<grid, block, 0, 0>>>(rows, cols_src, cols_dst, correspondance_cost, correspondance_device, valid_device); */

    CHECK(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(result, patch_similarity, rows * cols_src * cols_dst * sizeof(*result), cudaMemcpyDeviceToHost));
	long count = 0;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols_src; j++) {
			for (int k = 0; k < cols_dst; k++) {
				double x = result2 I(i,j,k);
				double y = result I(i,j,k);
				if (fabs(x - y) >= 0.00001) {
					count++;
				}
			}
		}
	}
	long total = rows * cols_src * cols_dst;
	printf("%ld / %ld = %g\n", count, total, (double)count / (double)total);
    cudaMemcpy(correspondance, correspondance_device, rows * cols_src * sizeof(*correspondance), cudaMemcpyDeviceToHost);
    cudaMemcpy(valid, valid_device, rows * cols_src * sizeof(*valid), cudaMemcpyDeviceToHost);

    cudaFree(pixel_similarity);
    cudaFree(patch_similarity);
    cudaFree(correspondance_device);
    cudaFree(valid_device);

    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return elapsed;
}

/*

extern "C" double
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double *src, double *dst, long *correspondance, char *valid) {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    if (ncuda_devices == 0) {
        return std::nan("");
    }
    cudaSetDevice(0);

    dim3 block_rsd(32, 1, 32);
    dim3 grid_rsd(
            (cols_src + block_rsd.x - 1) / block_rsd.x, 
            (rows + block_rsd.y - 1) / block_rsd.y, 
            (cols_dst + block_rsd.z - 1) / block_rsd.z);
    dim3 block_sd(1, 32, 32);
    dim3 grid_sd(
            (cols_src + block_sd.x - 1) / block_sd.x, 
            1, 
            (cols_dst + block_sd.z - 1) / block_sd.z);
    dim3 block_r(1024, 1, 1);
    dim3 grid_r(
            1, 
            (rows + block_r.y - 1) / block_r.y, 
            1);

    dim3 block_rs(32, 32, 1);
    dim3 grid_rs(
            (cols_src + block_rs.x - 1) / block_rs.x, 
            (rows + block_rs.y - 1) / block_rs.y, 
            1);
    dim3 block_rd(1, 32, 32);
    dim3 grid_rd(
            1, 
            (rows + block_rd.y - 1) / block_rd.y, 
            (cols_dst + block_rd.z - 1) / block_rd.z);

    double *src_device, *dst_device;
    cudaMalloc(&src_device, rows * cols_src * sizeof(*src_device));
    cudaMalloc(&dst_device, rows * cols_dst * sizeof(*dst_device));
    cudaMemcpy(src_device, src, rows * cols_src * sizeof(*src_device), cudaMemcpyHostToDevice);
    cudaMemcpy(dst_device, dst, rows * cols_dst * sizeof(*dst_device), cudaMemcpyHostToDevice);

    double *pixel_similarity, *patch_similarity, *correspondance_cost;
    cudaMalloc(&pixel_similarity, rows * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&patch_similarity, rows * cols_src * cols_dst * sizeof(double));
    cudaMalloc(&correspondance_cost, rows * cols_src * cols_dst * sizeof(double));
    long *correspondance_device; 
    cudaMalloc(&correspondance_device, rows * cols_src * sizeof(long));
    char *valid_device;
    cudaMalloc(&valid_device, rows * cols_src * sizeof(char));

    auto start = std::chrono::high_resolution_clock::now(); 

    get_pixel_similarity<<<grid_rsd, block_rsd, 0, 0>>>(rows, cols_src, cols_dst, src_device, dst_device, pixel_similarity);
    cumulative_sum_rows<<<grid_sd, block_sd, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    cumulative_sum_cols_src<<<grid_rs, block_rs, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);
    cumulative_sum_cols_dst<<<grid_rd, block_rd, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    get_patch_similarity<<<grid_rsd, block_rsd, 0, 0>>>(rows, cols_src, cols_dst, patch_size, pixel_similarity, patch_similarity);
    find_correspondances<<<grid_rsd, block_rsd, 0, 0>>>(rows, cols_src, cols_dst, patch_similarity, correspondance_cost);
    traceback_correspondance<<<grid_r, block_r, 0, 0>>>(rows, cols_src, cols_dst, correspondance_cost, correspondance_device, valid_device);

    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();

    cudaMemcpy(correspondance, correspondance_device, rows * cols_src * sizeof(long), cudaMemcpyDeviceToHost);
    cudaMemcpy(valid, valid_device, rows * cols_src * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(pixel_similarity);
    cudaFree(patch_similarity);
    cudaFree(correspondance_device);
    cudaFree(valid_device);

    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return elapsed;
}

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
*/


/*
    cumulative_sum_cols<<<grid_rd2, block_rd2, 0, 0>>>(rows, cols_src, cols_dst, pixel_similarity);

    dim3 block_rd2(32, 1, 32);
    dim3 grid_rd2(
            1, 
            (rows + block_rd2.y - 1) / block_rd2.y, 
            (cols_src + cols_dst + block_rd2.z - 1) / block_rd2.z);
__global__ void 
cumulative_sum_cols(long rows, long cols_src, long cols_dst, double *array) {
    long r = blockIdx.y * blockDim.y + threadIdx.y;
    long d = blockIdx.z * blockDim.z + threadIdx.z;

    //printf("return %ld %ld\n", r, d);
    if (r >= rows || d >= cols_src + cols_dst) return;

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
*/


//////////////////////////// SAMPLE CODE //////////////////////////////////////////

__global__ void 
matrix_vector_kernel(long rows, long cols, double *matrix, double *vector, double *result) {
    extern __shared__ double shared_memory[];

    long r = blockIdx.y;
    long s = blockIdx.x * blockDim.x + threadIdx.x;

    double warp_sum = s >= cols ? 0.0 : matrix[r * cols + s] * vector[s];

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

