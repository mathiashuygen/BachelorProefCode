#include <iostream>
#include <cuda_runtime.h>
#include <thread>

// Kernel that performs a simple computation
__global__ void maxUtilizationKernel(float* output, int n, int loopDuration)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Some computations to keep the SM busy
        float value = 0.0f;
        for (int i = 0; i < loopDuration; i++) {
            value += sinf(tid * 0.1f + i) * cosf(tid * 0.1f);
        }
        output[tid] = value;
    }
}



class thread_obj{
  public: 
    void operator()(int totalBlocks, int threadsPerBlock, int n, int loopDuration, int threadID){

      cudaStream_t threadStream;
      cudaStreamCreate(&threadStream);

      float* d_output;
      cudaMalloc(&d_output, n * sizeof(float));

      maxUtilizationKernel<<<totalBlocks, threadsPerBlock, 0, threadStream>>>(d_output, n, loopDuration);
      cudaFree(d_output);
      cudaStreamSynchronize(threadStream);
      cudaStreamDestroy(threadStream);
      std::cout<<"done executing kernel in thread: "<<threadID <<"\n";

    }
};


int main()
{
    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
   
    cudaStream_t busyStream1, busyStream2;
    
    cudaStreamCreate(&busyStream1);
    cudaStreamCreate(&busyStream2);



    // Choose grid and block sizes to maximize SM utilization
    const int threadsPerBlock = 256;
    const int blocksPerSM = 32; // RTX 4070 Ti can handle multiple blocks per SM
    const int totalBlocks = deviceProp.multiProcessorCount * blocksPerSM;
    
    const int n = totalBlocks * threadsPerBlock;
  
    //spawn threads 
    std::thread thread1(thread_obj(), totalBlocks, threadsPerBlock, n, 1000000, 1);
    std::thread thread2(thread_obj(), totalBlocks, threadsPerBlock, n, 10, 2);
    

    thread1.join();
    thread2.join();

    // Wait for the kernel to finish
    
    // Clean up
    cudaDeviceReset();
    
    printf("Kernel execution completed\n");
    
    return 0;
}
