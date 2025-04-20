#ifndef MATRIX_MUL_JOB_LAUNCH_INFORMATION_H
#define MATRIX_MUL_JOB_LAUNCH_INFORMATION_H

#include "baseLaunchInformation.h"
#include <cuda_runtime.h>
#include <iterator>

class MatrixMulJobLaunchInfo : public KernelLaunchInfoBase {
public:
  MatrixMulJobLaunchInfo(Job *job) : KernelLaunchInfoBase(job) {}

  ~MatrixMulJobLaunchInfo() override;
};

#endif
