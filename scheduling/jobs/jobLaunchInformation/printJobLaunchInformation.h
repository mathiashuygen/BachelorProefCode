#ifndef PRINT_JOB_LAUNCH_INFORMATION_H
#define PRINT_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class PrintJobLaunchInfo : public KernelLaunchInfoBase {
public:
  cudaStream_t kernelStream;
  float *devicePtr;
  float *hostPtr;
  size_t size;
  int id;
  PrintJobLaunchInfo(Job *job, cudaStream_t stream, float *dptr, float *hptr,
                     size_t size, int id)
      : KernelLaunchInfoBase(job), kernelStream(stream), devicePtr(dptr),
        hostPtr(hptr), size(size), id(id) {}

  ~PrintJobLaunchInfo() override = default;

  void cleanup() override;
};

#endif
