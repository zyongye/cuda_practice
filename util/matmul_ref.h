#pragma once

#include <math.h>

void matmul_ref(float* A, float* B, float* C, int ds){
    for(int i = 0; i < ds; i++){
        for(int j = 0; j < ds; j++){
            float temp = 0;
            for(int k = 0; k < ds; k++){
                temp += A[i * ds + k] * B[k * ds + j];
            }
            C[i * ds + j] = temp;
        }
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



