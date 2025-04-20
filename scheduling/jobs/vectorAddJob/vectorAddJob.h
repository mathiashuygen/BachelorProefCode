#ifndef VECTOR_ADD_JOB_H
#define VECTOR_ADD_JOB_H

#include "../../schedulers/asyncCompletionQueue/completionQueue.h"
#include "../jobBase/job.h"
#include "../jobLaunchInformation/vectorAddJobLauncInformation.h"
#include "../kernels/vectorAdd.h"
#include <cstddef>
#include <cuda_runtime.h>

class VectorAddJob : public Job {

private:
  // callback that is envoked at the end of each kernel execution.
  static void CUDART_CB vectorAddKernelCallback(cudaStream_t stream,
                                                cudaError_t status, void *data);

  // callback constructor.
  static void addVectorAddKernelCallback(Job *job, cudaStream_t stream,
                                         float *d_A, float *d_B, float *d_C,
                                         float *A, float *B, float *C);

  int vectorSize = 0;
  float *d_A;
  float *d_B;
  float *d_C;
  // host pointers
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;
  cudaStream_t kernelStream;
  size_t nrOfElements;

public:
  // job definition that goes with a task.
  void execute() override;

  VectorAddJob(int threadsPerBlock, int vectorSize);
  ~VectorAddJob() override;

  std::string getMessage() override;
};

#endif
