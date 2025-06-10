#pragma once

template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>  // the number of element each thread now processing is TMxTN
__global__ void _sgemm5(
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
    const uint thread_num = (BM * BN) / (TM * TN);

    // calculate output matrix coordinate within one tile
    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    __shared__ float _As[BM][BK];
    __shared__ float _Bs[BK][BN];

    // advance pointers to the starting positions
    A += cRow * BM * _K;
    B += cCol * BN;
    C += cRow * BM * _N + cCol * BN;

    // calculate shared memory index
    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint A_stride = thread_num / BK;

    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;
    const uint B_stride = thread_num / BN;

    float threadResults[TM][TN] = {0.};
    float regM[TM] = {0.};
    float regN[TN] = {0.};

    #pragma unroll
    for(int k = 0; k < _K; k += BK){
        // first load all block into shared mem
        for(int i = 0; i < BM; i += A_stride){
            _As[i + innerRowA][innerColA] = A[(i + innerRowA) * _K + innerColA];
        }

        for(int i = 0; i < BK; i += B_stride){
            _Bs[i + innerRowB][innerColB] = B[(i + innerRowB) * _N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * _N;

        // now one thread is managing TMxTN outputs entries
        #pragma unroll
        for(size_t dotIdx = 0; dotIdx < BK; ++dotIdx){
            for(int i = 0; i < TM; i++){
                regM[i] = _As[threadRow * TM + i][dotIdx];
            }
            for(int i = 0; i < TN; i++){
                regN[i] = _Bs[dotIdx][threadCol * TN + i];
            }
            for(int i = 0; i < TM; i++){
                for(int j = 0; j < TN; j++){
                    threadResults[i][j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for(int i = 0; i < TM; i++){
        for(int j = 0; j < TN; j++){
            C[(threadRow * TM + i) * _N + (threadCol * TN + j)] = alpha * threadResults[i][j] + beta * C[(threadRow * TM + i) * _N + (threadCol * TN + j)];
        }
    }
}

void sgemm5(
    float *d_C,
    const float *d_A,
    const float *d_B,
    const float alpha,
    const float beta,
    const int _M, const int _N, const int _K
){
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 gridDim(CEIL_DIV(_N, BN), CEIL_DIV(_M, BM));
    // again, each threadblock is responsible for one tile output,
    // but now since one thread is now mamaged for 3 output entries,
    // the total thread per threadblock decreases
    dim3 blockDim(16 * 16);

    _sgemm5<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(d_C, d_A, d_B, alpha, beta, _M, _N, _K);
    cudaCheckErrors("Matmul fails");
}

