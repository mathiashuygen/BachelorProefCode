#include <stdio.h>
#include <cuda_runtime.h>
#include "libsmctrl.h"

// Kernel that performs a simple computation
__global__ void maxUtilizationKernel(float* output, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Some computations to keep the SM busy
        float value = 0.0f;
        for (int i = 0; i < 1000; i++) {
            value += sinf(tid * 0.1f + i) * cosf(tid * 0.1f);
        }
        output[tid] = value;
    }
}

int main()
{
    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Running on GPU: %s\n", deviceProp.name);
    printf("Number of SMs: %d\n", deviceProp.multiProcessorCount);
    
    cudaStream_t busyStream;
    cudaStreamCreate(&busyStream);

      //set a mask for the stream, alternating mask which means 50% of the tpcs are used.
    libsmctrl_set_stream_mask((void*)busyStream, 0x1);
    // Choose grid and block sizes to maximize SM utilization
    const int threadsPerBlock = 256;
    const int blocksPerSM = 32; // RTX 4070 Ti can handle multiple blocks per SM
    const int totalBlocks = deviceProp.multiProcessorCount * blocksPerSM;
    
    const int n = totalBlocks * threadsPerBlock;
    printf("Launching kernel with %d blocks, %d threads per block (%d total threads)\n", 
           totalBlocks, threadsPerBlock, n);
    
    // Allocate memory
    float* d_output;
    cudaMalloc(&d_output, n * sizeof(float));
    
    // Launch kernel
    maxUtilizationKernel<<<totalBlocks, threadsPerBlock, 0, busyStream>>>(d_output, n);
    
    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // Clean up
    cudaFree(d_output);
    cudaDeviceReset();
    
    printf("Kernel execution completed\n");
    
    return 0;
}
