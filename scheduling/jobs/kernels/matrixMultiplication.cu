#include "matrixMultiplication.h"
// Introduction to GPU Kernels and Hardware, Richard Ansorge p58.
// DOI: https://doi.org/10.1017/9781108855273.002
__global__ void matrixMul(float *A, float *B, float *C, int matrixDim) {
  int tix = blockIdx.x * blockDim.x + threadIdx.x;
  int tiy = blockIdx.y * blockDim.y + threadIdx.y;

  // bound check
  if (tiy > matrixDim || tix > matrixDim)
    return;

  C[tiy * matrixDim + tix] = 0.0;

  for (int k = 0; k < matrixDim; k++) {
    C[tiy * matrixDim + tix] += A[tiy * matrixDim + k] * B[k * matrixDim + tix];
  }
}
