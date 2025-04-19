#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <cuda_runtime.h>

__global__ void vectorAddKernel(float *a, float *b, float *c, int N);

#endif
