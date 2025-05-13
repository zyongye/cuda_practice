#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#define VECTOR_SIZE (1024 * 1024 + 1)
#define BLOCK_SIZE 512

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

__global__ void vector_add(float* A, float* B, float* C, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < N)
        C[index] = A[index] + B[index];
}

float A[VECTOR_SIZE];
float B[VECTOR_SIZE];
float C[VECTOR_SIZE];

int main(){

    for(int i = 0; i < VECTOR_SIZE; i++){
        A[i] = rand() / (float) RAND_MAX;
        B[i] = rand() / (float) RAND_MAX;
    }

    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_B, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_C, VECTOR_SIZE * sizeof(float));
    cudaCheckErrors("malloc failure");

    cudaMemcpy(d_A, A, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("memcpy failure");

    vector_add<<<(VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, VECTOR_SIZE);
    cudaCheckErrors("addition fail");

    cudaMemcpy(C, d_C, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy back failure");

    printf("A[0] = %.4f\n", A[0]);
    printf("B[0] = %.4f\n", B[0]);
    printf("C[0] = %.4f\n", C[0]);

}


