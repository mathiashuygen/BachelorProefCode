#include "printJobLaunchInformation.h"
#include <iostream>
void PrintJobLaunchInfo::cleanup() {
  std::cout << "delete print job launch info\n";
  cudaStreamSynchronize(kernelStream);
  cudaFreeHost(hostPtr);
  cudaFree(devicePtr);
  cudaStreamDestroy(kernelStream);
}
