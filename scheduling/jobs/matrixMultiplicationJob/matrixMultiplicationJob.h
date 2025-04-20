#ifndef MATRIX_MUL_JOB_H
#define MATRIX_MUL_JOB_H

#include "../../schedulers/asyncCompletionQueue/completionQueue.h"
#include "../jobBase/job.h"
#include "../jobLaunchInformation/matrixMutliplicationJobLaunchInformation.h"
#include "../kernels/matrixMultiplication.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

class MatrixMultiplicationJob : public Job {

private:
  static void CUDART_CB matrixMulKernelCallback(cudaStream_t stream,
                                                cudaError_t status, void *data);
  static void addMatrixMulKernelCallback(Job *job, cudaStream_t stream);
  int matrixDim;
  float *d_A;
  float *d_B;
  float *d_C;
  // host pointers
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  cudaStream_t kernelStream;
  size_t nrOfElements;
  dim3 blocks;

public:
  virtual void execute() override;

  MatrixMultiplicationJob(int threadsPerBlock, int matrixDim);
  ~MatrixMultiplicationJob() override;

  std::string getMessage() override;
};

#endif
