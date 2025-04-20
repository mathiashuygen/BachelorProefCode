#ifndef BUSY_JOB_LAUNCH_INFORMATION_H
#define BUSY_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class BusyJobLaunchInfo : public KernelLaunchInfoBase {
public:
  BusyJobLaunchInfo(Job *job)

      : KernelLaunchInfoBase(job) {}

  ~BusyJobLaunchInfo() override;
};

#endif
