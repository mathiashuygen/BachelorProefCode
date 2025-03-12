#include <iostream>
#include <cuda_runtime.h>
#include "thrust/device_vector.h"



// Kernel that performs a simple that is memory intensive. It sums up all the elements of 4 arrays. These arrays are stored in the GPU memory. Reading and writing are
// memory bound tasks. So if a kernel reads and writes a lot, the kernel is memory intensive. 
__global__ void highMemoryUsage(float* A, float* B, float* C, float* D, long N){
  int tid = blockIdx.x * gridDim.x + threadIdx.x;

  while(tid < N){
    float aEl = A[tid];
    float bEl = B[tid];
    float cEl = C[tid];
    float dEl = D[tid];

    float total = aEl + bEl + cEl + dEl;

    D[tid] = total;
    tid += gridDim.x * blockDim.x;

  }

} 


