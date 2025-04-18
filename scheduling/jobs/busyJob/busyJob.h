#ifndef BUSY_JOB_H
#define BUSY_JOB_H

#include "../jobBase/job.h"
#include "../kernels/busyKernel.h"
#include <cuda_runtime.h>
#include <iostream>

class BusyJob : public Job {
private:
  struct busyKernelLaunchInformation {
    Job *jobPtr;
    cudaStream_t kernelStream; // stream in which the kernel is launched.
    float *devicePtr;          // Device memory pointer
    float *timerDptr; // device ptr to float that holds the total execution time
                      // of a kernel.
    float *hostPtr;   // Host memory pointer
    size_t size;      // Size of data to copy in bytes
    int taskId;       // id of  the task invoking jobs.

    busyKernelLaunchInformation(Job *job, cudaStream_t stream, float *dptr,
                                float *hptr, size_t sz, int id)
        : jobPtr(job), kernelStream(stream), devicePtr(dptr), hostPtr(hptr),
          size(sz), taskId(id) {}
  };

  // callback that is envoked at the end of each kernel execution.
  static void CUDART_CB busyKernelCallback(cudaStream_t stream,
                                           cudaError_t status, void *data);

  // callback constructor.
  static void addBusyKernelCallback(Job *job, cudaStream_t stream, float *dptr,
                                    float *hptr, size_t size, int id);

public:
  // job definition that goes with a task.
  void execute() override;

  BusyJob(int threadsPerBlock, int threadBlocks);

  std::string getMessage() override;
};

#endif
