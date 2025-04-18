
#ifndef BUSY_KERNEL_H
#define BUSY_KERNEL_H

#include <cuda_runtime.h>

__global__ void maxUtilizationKernel(float *output, int duration);

#endif
