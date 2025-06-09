#pragma once

__global__ void _sgemm3(
    float *C,
    const float *A,
    const float *B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    const size_t BLOCK_SIZE = 32;

    // make adjanct warp loading adjanct entry, memory coaloasing
    const uint x = threadIdx.x / BLOCK_SIZE;
    const uint y = threadIdx.x % BLOCK_SIZE;

    __shared__ float _As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float _Bs[BLOCK_SIZE][BLOCK_SIZE];

    // advance pointers to the starting positions
    A += blockIdx.x * BLOCK_SIZE * _K;
    B += blockIdx.y * BLOCK_SIZE;
    C += blockIdx.x * BLOCK_SIZE * _N + blockIdx.y * BLOCK_SIZE;

    float tmp = 0.0;

    for(int k = 0; k < _K; k+=BLOCK_SIZE){
        // first load all BLOCK_SIZE * BLOCK_SIZE into shared mem
        _As[x][y] = A[x * _K + y];
        _Bs[x][y] = B[x * _N + y];

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * _N;

        // then each threads calculate partial sum of available entry 
        // that are in the share mem
        for(size_t dotIdx = 0; dotIdx < BLOCK_SIZE; dotIdx++)
            tmp += _As[x][dotIdx] * _Bs[dotIdx][y];

        __syncthreads();
    }

    C[x * _N + y] = alpha * tmp + beta * C[x * _N + y];
}

void sgemm3(
    float *d_C,
    const float *d_A,
    const float *d_B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    dim3 blockDim(32 * 32);
    dim3 gridDim(CEIL_DIV(_M, 32), CEIL_DIV(_N, 32));

    _sgemm3<<<gridDim, blockDim>>>(d_C, d_A, d_B, alpha, beta, _M, _N, _K);
    cudaCheckErrors("Matmul fails");
}

