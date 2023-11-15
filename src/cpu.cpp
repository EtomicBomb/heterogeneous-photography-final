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

extern "C" double 
matrix_vector(int problem_count, int rows, int cols, double *matrix, double *vector, double *result) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < problem_count; i++) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[i * rows + r] += matrix[i * rows * cols + r * cols + c] * vector[i * cols + c];
            }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    return elapsed / problem_count;
}

