#ifndef BUSY_JOB_LAUNCH_INFORMATION_H
#define BUSY_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class BusyJobLaunchInfo : public KernelLaunchInfoBase {
public:
  cudaStream_t kernelStream;
  float *devicePtr;
  float *hostPtr;
  float *timerptr;
  size_t size;
  int id;
  BusyJobLaunchInfo(Job *job, cudaStream_t stream, float *dptr, float *hptr,
                    size_t size, int id)
      : KernelLaunchInfoBase(job), kernelStream(stream), devicePtr(dptr),
        hostPtr(hptr), size(size), id(id) {}

  ~BusyJobLaunchInfo() override = default;

  void cleanup() override;
};

#endif
