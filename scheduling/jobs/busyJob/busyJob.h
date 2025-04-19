#ifndef BUSY_JOB_H
#define BUSY_JOB_H

#include "../../schedulers/asyncCompletionQueue/completionQueue.h"
#include "../jobBase/job.h"
#include "../jobLaunchInformation/busyJobLaunchInformation.h"
#include "../kernels/busyKernel.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <iterator>

class BusyJob : public Job {
private:
  // callback that is envoked at the end of each kernel execution.
  static void CUDART_CB busyKernelCallback(cudaStream_t stream,
                                           cudaError_t status, void *data);

  // callback constructor.
  static void addBusyKernelCallback(Job *job, cudaStream_t stream, float *dptr,
                                    float *hptr, size_t size, int id);
  size_t nrOfBytes;
  float *hptr;
  float *dptr;
  float *timerptr;

  cudaStream_t busyStream;

public:
  // job definition that goes with a task.
  void execute() override;

  BusyJob(int threadsPerBlock, int threadBlocks);

  ~BusyJob() override;

  std::string getMessage() override;
};

#endif
