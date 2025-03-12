#include <iostream>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "libsmctrl.h"

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void firstKernel(float *A, float *B, float *C, int N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidx < N && tidy < N) {
        int index = tidy * N + tidx; // Linearize 2D indices
        C[index] = A[index] + B[index];
    }
}

int main() {
    int N = 800;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
  
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    libsmctrl_set_stream_mask((void*)stream, 0xFFFFFFFFFFFFC00);

    
    thrust::host_vector<float> A(N*N);
    thrust::host_vector<float> B(N*N);
    thrust::host_vector<float> C(N*N);
    
    thrust::host_vector<float> D(N*N);
    thrust::host_vector<float> E(N*N);
    thrust::host_vector<float> F(N*N);
    

    // Initialize matrices A, B, D and E
    for (int i = 0; i < N*N; i++) {
      A[i] = (float)i;
      B[i] = (float)i + 1.0;
      D[i] = (float)i;
      E[i] = (float)i + 1.0;
    }


    size_t size = N * N * sizeof(float);

    // Allocate device memory
    thrust::device_vector<float>d_A(size);
    thrust::device_vector<float>d_B(size);
    thrust::device_vector<float>d_C(size);
    d_A = A; 
    d_B = B;
   
  

    thrust::device_vector<float>d_D(size);
    thrust::device_vector<float>d_E(size);
    thrust::device_vector<float>d_F(size);
    d_D = D;
    d_E = E;
    

    float *d_aptr = thrust::raw_pointer_cast(&d_A[0]);
    float *d_cptr = thrust::raw_pointer_cast(&d_B[0]);
    float *d_bptr = thrust::raw_pointer_cast(&d_C[0]);

    float *d_dptr = thrust::raw_pointer_cast(&d_D[0]);
    float *d_eptr = thrust::raw_pointer_cast(&d_E[0]);
    float *d_fptr = thrust::raw_pointer_cast(&d_F[0]);





    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);  // 256 threads total, arranged in 16x16
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    // Launch the kernel
    firstKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_aptr, d_bptr, d_cptr, N);

    
    //wait for gpu to finish executing
    cudaDeviceSynchronize();
    

    //copy result from device memory to host memory.
    C = d_C; 


    // Verify the result
    bool success = true;
    for (int i = 0; i < N*N; i++) {
            float expected = A[i] + B[i];
            float expected2 = D[i] + E[i];
            if (C[i] != expected) {
                printf("Error at C[%d]: expected %.2f, got %.2f\n", i, expected, C[i]);
                success = false;
                break;
            } 
        
        if (!success) break;
    }
    if (success) {
        printf("Matrix addition successful!\n");
    }

    cudaStreamDestroy(stream);



    return 0;
}
