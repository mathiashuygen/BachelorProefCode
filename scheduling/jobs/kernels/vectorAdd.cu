#include "vectorAdd.h"

__global__ void vectorAddKernel(float *a, float *b, float *c, int N) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx < N) {
    c[tidx] = a[tidx] + b[tidx];
  }
}
