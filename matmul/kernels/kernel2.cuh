#pragma once



__global__ void _sgemm2(
    float *C,
    const float *A,
    const float *B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    const int BLOCK_SIZE = 32;
    // change work assignment such that adjanct threads are accessing the same row in A,
    // and consecitive column entry in B
    // therefore the memory access of A is boardcast and access to B can be coalesces
    const uint y = blockIdx.x * blockDim.x + threadIdx.x;
    const uint x = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < _M && y < _N){
        float tmp = 0.0;
        for(int k = 0; k < _K; k++){
            // consecitive threads have cansecutive y value
            tmp += A[x * _K + k] * B[k * _N + y];
        }
        C[x * _N + y] = alpha * tmp + beta * C[x * _N + y];
    }
}

void sgemm2(
    float *d_C,
    const float *d_A,
    const float *d_B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    dim3 blockDim(32, 32);
    dim3 gridDim(CEIL_DIV(_M, 32), CEIL_DIV(_N, 32));

    _sgemm2<<<gridDim, blockDim>>>(d_C, d_A, d_B, alpha, beta, _M, _N, _K);
    cudaCheckErrors("Matmul fails");
}





