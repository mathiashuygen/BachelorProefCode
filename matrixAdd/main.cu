#include <stdio.h>
#include <cuda_runtime.h>
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
    int N = 570;
    int numberOfStreams = 2;
    
    cudaStream_t streams[numberOfStreams];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    
    const uint64_t masks[2] = {0x1, 0x3};

    //libsmctrl_set_stream_mask((void*)streams[0], masks[0]);
    
    float A[N][N], B[N][N], C[N][N];
    float D[N][N], E[N][N], F[N][N];
    

    // Initialize matrices A, B, D and E
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)i;
            B[i][j] = (float)i + 1.0;
            D[i][j] = (float)i;
            E[i][j] = (float)i + 1.0;
        }
    }

    float *d_A, *d_B, *d_C;
    float *d_D, *d_E, *d_F;
    

    size_t size = N * N * sizeof(float);

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, size));

    CHECK_CUDA_ERROR(cudaMalloc(&d_D, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_E, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_F, size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemcpy(d_D, D, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_E, E, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    firstKernel<<<gridSize, blockSize, 0, streams[0]>>>(d_A, d_B, d_C, N);
    //firstKernel<<<gridSize, blockSize, 0, streams[1]>>>(d_D, d_E, d_F, N);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaMemcpy(F, d_F, size, cudaMemcpyDeviceToHost));

    // Verify the result
    bool success = true;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float expected = A[i][j] + B[i][j];
            float expected2 = D[i][j] + E[i][j];
            if (C[i][j] != expected) {
                printf("Error at C[%d][%d]: expected %.2f, got %.2f\n", i, j, expected, C[i][j]);
                success = false;
                break;
            } 
        }
        if (!success) break;
    }
    if (success) {
        printf("Matrix addition successful!\n");
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    CHECK_CUDA_ERROR(cudaFree(d_D));
    CHECK_CUDA_ERROR(cudaFree(d_E));
    CHECK_CUDA_ERROR(cudaFree(d_F));

    return 0;
}
