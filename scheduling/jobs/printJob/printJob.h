#ifndef PRINT_JOB_H
#define PRINT_JOB_H

#include "../../schedulers/asyncCompletionQueue/completionQueue.h"
#include "../jobBase/job.h"
#include "../jobLaunchInformation/printJobLaunchInformation.h"
#include "../kernels/printKernel.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

class PrintJob : public Job {

private:
  static void CUDART_CB printKernelCallback(cudaStream_t stream,
                                            cudaError_t status, void *data);
  static void addPrintKernelCallback(Job *job, cudaStream_t stream, float *dptr,
                                     float *hptr, size_t size, int id);
  size_t nbrOfBytes;
  float *hostPtr = nullptr;
  float *dptr;
  cudaStream_t jobStream;

public:
  virtual void execute() override;

  PrintJob(int threadsPerBlock, int threadBlocks);
  ~PrintJob() override;

  std::string getMessage() override;
};

#endif
