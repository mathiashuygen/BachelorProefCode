#ifndef PRINT_JOB_LAUNCH_INFORMATION_H
#define PRINT_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class PrintJobLaunchInfo : public KernelLaunchInfoBase {
public:
  PrintJobLaunchInfo(Job *job) : KernelLaunchInfoBase(job) {}

  ~PrintJobLaunchInfo() override;
};

#endif
