#include <iostream>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    const int m = 512, n = 512, k = 512;  // Larger size to highlight Tensor Core usage

    float *A_fp32 = new float[m * k];
    float *B_fp32 = new float[k * n];
    float *C_fp32 = new float[m * n]();

    for (int i = 0; i < m * k; ++i) A_fp32[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) B_fp32[i] = 1.0f;

    __nv_bfloat16 *A_bf16 = new __nv_bfloat16[m * k];
    __nv_bfloat16 *B_bf16 = new __nv_bfloat16[k * n];
    for (int i = 0; i < m * k; ++i) A_bf16[i] = __float2bfloat16(A_fp32[i]);
    for (int i = 0; i < k * n; ++i) B_bf16[i] = __float2bfloat16(B_fp32[i]);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, m * k * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, k * n * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_C, m * n * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, A_bf16, m * k * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B_bf16, k * n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, C_fp32, m * n * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;

    // --- Profiling ---
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Tensor Core GEMM
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha,
        d_B, CUDA_R_16BF, n,
        d_A, CUDA_R_16BF, k,
        &beta,
        d_C, CUDA_R_32F, n,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Tensor Core path
    ));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::cout << "Tensor Core BF16 GEMM time: " << elapsed_ms << " ms\n";

    // Optionally copy back and check a few values
    CHECK_CUDA(cudaMemcpy(C_fp32, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "C[0] = " << C_fp32[0] << " (should be close to " << k << ")\n";

    // Cleanup
    delete[] A_fp32;
    delete[] B_fp32;
    delete[] C_fp32;
    delete[] A_bf16;
    delete[] B_bf16;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}