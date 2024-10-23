#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// 该模板函数使用一维块平铺的方法实现矩阵乘法
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
  const float* A, const float* B, float beta,
  float* C) {

  //如果我们在这里翻转 x 和 y，则大型矩阵的性能会降低 ~30%。当前快了 30% 的配置确保具有顺序 blockID 的 block 可以按顺序访问 B 的列，同时共享同一行 A 。较慢的配置将共享 A 的列，但对 B 的访问将是非顺序的。因此，更快的配置具有更好的空间局部性，因此 L2 命中率更高。

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 计算每个warp将处理的元素数量，每个warp将计算32*TM元素，32为列维度的大小。
  const int threadCol = threadIdx.x % BN;
  // 计算当前线程在块中的行索引。
  const int threadRow = threadIdx.x / BN;

  // 为 SMEM 中的当前 blocktile 分配空间
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  //待办事项：调整到每个线程以加载多个条目，并更好地利用缓存大小
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  // 在registerfile中为结果分配线程本地缓存
  float threadResults[TM] = { 0.0 };

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 我们将dotproduct循环设置为外部循环，以便重用Bs条目，我们可以将其缓存到tmp var中。
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
      alpha * threadResults[resIdx] +
      beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}