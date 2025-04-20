#ifndef VECTOR_ADD_JOB_LAUNCH_INFORMATION_H
#define VETOR_ADD_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class VectorAddJobLaunchInfo : public KernelLaunchInfoBase {
public:
  VectorAddJobLaunchInfo(Job *job) : KernelLaunchInfoBase(job) {}

  ~VectorAddJobLaunchInfo() override;
};

#endif
