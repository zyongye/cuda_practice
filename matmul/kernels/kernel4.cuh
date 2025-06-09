#pragma once

template<const int BM,
         const int BN,
         const int BK,
         const int TM>  // the number of element each thread now processing
__global__ void _sgemm4(
    float *C,
    const float *A,
    const float *B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    // decouple physical block index with logical block
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // calculate output matrix coordinate within one tile
    const uint threadCol = threadIdx.x % BN;
    const uint threadRow = threadIdx.x / BN;

    __shared__ float _As[BM][BK];
    __shared__ float _Bs[BK][BN];

    // advance pointers to the starting positions
    A += cRow * BM * _K;
    B += cCol * BN;
    C += cRow * BM * _N + cCol * BN;

    // calculate shared memory index
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.};

    #pragma unroll
    for(int k = 0; k < _K; k += BK){
        // first load all block into shared mem
        _As[innerRowA][innerColA] = A[innerRowA * _K + innerColA];
        _Bs[innerRowB][innerColB] = B[innerRowB * _N + innerColB];

        __syncthreads();

        A += BK;
        B += BK * _N;

        // now one thread is managing TM outputs entries
        #pragma unroll
        for(size_t dotIdx = 0; dotIdx < BK; ++dotIdx){
            float tmpB = _Bs[dotIdx][threadCol];
            #pragma unroll
            for(int resIdx = 0; resIdx < TM; ++resIdx){
                threadResults[resIdx] += _As[threadRow * TM + resIdx][dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; i++){
        C[(threadRow * TM + i) * _N + threadCol] = alpha * threadResults[i] + beta * C[(threadRow * TM + i) * _N + threadCol];
    }
}

void sgemm4(
    float *d_C,
    const float *d_A,
    const float *d_B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8;
    dim3 gridDim(CEIL_DIV(_N, BN), CEIL_DIV(_M, BM));
    // again, each threadblock is responsible for one tile output,
    // but now since one thread is now mamaged for 3 output entries,
    // the total thread per threadblock decreases
    dim3 blockDim((BM * BN) / TM);

    _sgemm4<BM, BN, BK, TM><<<gridDim, blockDim>>>(d_C, d_A, d_B, alpha, beta, _M, _N, _K);
    cudaCheckErrors("Matmul fails");
}

