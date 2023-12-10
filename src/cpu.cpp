#include <stdio.h>
#include <complex>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <sched.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <omp.h>

#define I(r, s, d) [(r) * cols_dst * cols_src + (s) * cols_dst + (d)]
#define Isrc(r, s) [(r) * cols_src + (s)]
#define Idst(r, d) [(r) * cols_dst + (d)]

inline long 
argmin3(double x, double y, double z) {
    if (x < y) {
        return x < z ? 0 : 2;
    } 
    return y < z ? 1 : 2;
}

inline double 
min3(double x, double y, double z) {
    long index = argmin3(x, y, z);
    double numbers[] = {x, y, z};
    return numbers[index];
}

void
get_pixel_similarity(long rows, long cols_src, long cols_dst, const double *src, const double *dst, double *pixel_similarity) {
    #pragma omp parallel for collapse(3)
    for (long r = 0; r < rows; r++) {
        for (long s = 0; s < cols_src; s++) {
            for (long d = 0; d < cols_dst; d++) {
                double distance = src Isrc(r, s) - dst Idst(r, d);
                pixel_similarity I(r, s, d) = distance * distance;
            }
        }
    }
}

inline void 
sum_diagonal(long rows, long cols_src, long cols_dst, double *array, long r, long s, long d) {
    s += 1;
    d += 1;
    while (s < cols_src && d < cols_dst) {
        array I(r, s, d) += array I(r, s - 1, d - 1);
        s += 1;
        d += 1;
    }
}

void 
cumulative_sum_cols_src(long rows, long cols_src, long cols_dst, double *array) {
    #pragma omp parallel for collapse(2)
    for (long r = 0; r < rows; r++) {
        for (long s = 1; s < cols_src; s++) { // ignore central diagonal
            long d = 0;
            sum_diagonal(rows, cols_src, cols_dst, array, r, s, d);
        }
    }
}

void 
cumulative_sum_cols_dst(long rows, long cols_src, long cols_dst, double *array) {
    #pragma omp parallel for collapse(2)
    for (long r = 0; r < rows; r++) {
        for (long d = 0; d < cols_dst; d++) {
            long s = 0;
            sum_diagonal(rows, cols_src, cols_dst, array, r, s, d);
        }
    }
}

void 
cumulative_sum_rows(long rows, long cols_src, long cols_dst, double *array) {
    #pragma omp parallel for collapse(2)
    for (long s = 0; s < cols_src; s++) {
        for (long d = 0; d < cols_dst; d++) {
            for (long r = 1; r < rows; r++) {
                array I(r, s, d) += array I(r - 1, s, d);
            }
        }
    }
}

void 
calculate_patch_similarity(long rows, long cols_src, long cols_dst, long patch_size, const double *pixel_similarity, double *patch_similarity) {
    #pragma omp parallel for collapse(3)
    for (long r = 0; r < rows; r++) {
        for (long s = 0; s < cols_src; s++) {
            for (long d = 0; d < cols_dst; d++) {
                long rm = r - patch_size - 1;
                long sm = s - patch_size - 1;
                long dm = d - patch_size - 1;

                long overflow = std::max(0l, std::max(s + patch_size - cols_src + 1, d + patch_size - cols_dst + 1));
                long underflow = - std::min(0l, 1 + std::min(sm, dm));

                long rp = std::min(r + patch_size, rows - 1);
                long sp = s + patch_size - overflow;
                long dp = d + patch_size - overflow;

                double rpsp = pixel_similarity I(rp, sp, dp);
                double rpsm = sm >= 0 && dm >= 0 ? pixel_similarity I(rp, sm, dm) : 0.0;
                double rmsp = rm >= 0 ? pixel_similarity I(rm, sp, dp): 0.0;
                double rmsm = rm >= 0 && sm >= 0 && dm >= 0 ? pixel_similarity I(rm, sm, dm) : 0.0;
                double sum = rpsp + rmsm - rpsm - rmsp;

                long width = rp - std::max(rm, -1l);
                long height = 2 * patch_size + 1 - underflow - overflow;
                long count = width * height;

                patch_similarity I(r, s, d) = sum / count;
            }
        }
    }
}

void
calculate_costs(long rows, long cols_src, long cols_dst, double occlusion_cost, const double *patch_similarity, double *cost) {
    #pragma omp parallel 
    {
        #pragma omp for collapse(2) nowait
        for (long r = 0; r < rows; r++) {
            for (long s = 0; s < cols_src; s++) {
                long d = 0;
                cost I(r, s, d) = s * occlusion_cost;
            }
        }
        #pragma omp for collapse(2)
        for (long r = 0; r < rows; r++) {
            for (long d = 0; d < cols_dst; d++) {
                long s = 0;
                cost I(r, s, d) = d * occlusion_cost;
            }
        }
        #pragma omp for
        for (long r = 0; r < rows; r++) {
            for (long k = 0; k < cols_src + cols_dst; k++) {
                long s_low = std::max(1l, k - cols_dst + 1);
                long s_high = std::min(cols_src, k);
                for (long s = s_low; s < s_high; s++) {
                    long d = k - s;
                    double match = cost I(r, s - 1, d - 1) + patch_similarity I(r, s, d);
                    double left = cost I(r, s, d - 1) + occlusion_cost;
                    double up = cost I(r, s - 1, d) + occlusion_cost;
                    cost I(r, s, d) = min3(match, left, up);
                }
            }
        }
    }
}

void
traceback_correspondence(long rows, long cols_src, long cols_dst, const double *cost, long *correspondence, char *valid) {
    #pragma omp parallel for 
    for (long r = 0; r < rows; r++) {
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
}

extern "C" int
scanline_stereo(long rows, long cols_src, long cols_dst, long patch_size, double occlusion_cost, const double *src, const double *dst, long *correspondence, char *valid, float *timings) {
    std::vector<double> pixel_similarity(rows * cols_src * cols_dst);
    std::vector<double> patch_similarity(rows * cols_src * cols_dst);
    std::vector<double> cost(rows * cols_src * cols_dst);
    std::memset(valid, 0, rows * cols_src * sizeof(*valid));

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point stop;
    start = std::chrono::high_resolution_clock::now();

    get_pixel_similarity(rows, cols_src, cols_dst, src, dst, pixel_similarity.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[1] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    cumulative_sum_rows(rows, cols_src, cols_dst, pixel_similarity.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[2] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    cumulative_sum_cols_src(rows, cols_src, cols_dst, pixel_similarity.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[3] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;
    
    cumulative_sum_cols_dst(rows, cols_src, cols_dst, pixel_similarity.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[4] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    calculate_patch_similarity(rows, cols_src, cols_dst, patch_size, pixel_similarity.data(), patch_similarity.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[5] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    calculate_costs(rows, cols_src, cols_dst, occlusion_cost, patch_similarity.data(), cost.data());

    stop = std::chrono::high_resolution_clock::now();
    timings[6] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    traceback_correspondence(rows, cols_src, cols_dst, cost.data(), correspondence, valid);

    stop = std::chrono::high_resolution_clock::now();
    timings[7] = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
    start = stop;

    return 0;
}

