#ifndef MATRIX_MUL_KERNEL_H
#define MATRIX_MUL_KERNEL_H
#include <cuda_runtime.h>

__global__ void matrixMul(float *A, float *B, float *C, int matrixDim);

#endif
