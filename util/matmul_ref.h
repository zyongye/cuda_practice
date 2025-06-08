#pragma once

#include <math.h>
#include <cstdlib>

#define CEIL_DIV(x, y)  (((x) + (y) - 1) / (y))

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

// shape of A: (M, K)
// shape of B: (K, N)
// shape of C: (M, N)
void matmul_ref(float* C, float* A, float* B, float alpha, float beta, int M, int N, int K){
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float temp = 0;
            for(int k = 0; k < K; k++){
                temp += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * temp + beta * C[i * N + j];
        }
    }
}

void init_rand(float* arr, size_t len){
    for (size_t i = 0; i < len; i++){
        arr[i] = (rand() - RAND_MAX / 2 ) / (float)RAND_MAX;
    }
}

bool assert_close(float* cuda_impl, float* ref, int len, float exp_max_err, float exp_rms_err){
    float max_error = 0.0f;
    float sum_squared_error = 0.0f;

    for (int i = 0; i < len; i++) {
        float err = fabsf(cuda_impl[i] - ref[i]);
        if (err > max_error) {
            max_error = err;
        }
        sum_squared_error += err * err;
    }

    float max_err = max_error;
    float rms_err = sqrtf(sum_squared_error / len);

    return (max_err <= exp_max_err) && (rms_err <= exp_rms_err);
}



