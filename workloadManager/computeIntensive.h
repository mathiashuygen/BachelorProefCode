#include <stdlib.h>
#include "cuda_runtime.h"




// Kernel that performs a simple that is compute intensive.
__global__ void maxUtilizationKernel(thrust::device_vector<float> output, int n, int iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Some computations to keep the SM busy
        float value = 0.0f;
        for (int i = 0; i < iterations; i++) {
            value += sinf(tid * 0.1f + i) * cosf(tid * 0.1f);
        }
        output[tid] = value;
    }
}


