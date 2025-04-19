#include "busyJobLaunchInformation.h"

void BusyJobLaunchInfo::cleanup() {
  cudaStreamSynchronize(this->kernelStream);

  cudaFreeHost(this->hostPtr);
  cudaFree(this->devicePtr);
  cudaFree(this->timerptr);
  cudaStreamDestroy(this->kernelStream);
}
