#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../util/matmul_ref.h"

#include "./kernels/kernel1.cuh"
#include "./kernels/kernel2.cuh"
#include "./kernels/kernel3.cuh"

#define M 4096
#define N 4096
#define K 4096

int main(){
    float *h_A, *h_B, *h_C, *h_C_ref;
    float *d_A, *d_B, *d_C, *d_C_ref;
    float alpha, beta;
    
    h_A = new float[M * K];
    h_B = new float[K * N];
    h_C = new float[M * N];
    h_C_ref = new float[M * N];

    init_rand(h_A, M * K);
    init_rand(h_B, K * N);
    init_rand(h_C, M * N);
    alpha = (rand() - RAND_MAX / 2 ) / (float)RAND_MAX;;
    beta = (rand() - RAND_MAX / 2 ) / (float)RAND_MAX;;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_C_ref, M * N * sizeof(float));
    cudaCheckErrors("Malloc fails");

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ref, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Memcpy fails");

    sgemm3(d_C, d_A, d_B, alpha, beta, M, N, K);
    cudaCheckErrors("Error executing kernel");

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back to host fails");

    printf("Kernel completes\n");

    // running reference implementation
    sgemm2(d_C_ref, d_A, d_B, alpha, beta, M, N, K);
    cudaCheckErrors("Error executing reference kernel");

    cudaMemcpy(h_C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Memcpy back to host fails");

    if(assert_close(h_C,h_C_ref, M * N, 0.02, 0.004)){
        printf("Success!\n"); 
    }else{
        printf("error!\n");
    }

    return 0;
}




