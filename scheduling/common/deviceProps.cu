#include "deviceProps.h"
#include <cmath>
#include <cstdint>

// init all the static vars for the linker, their actual value will be set
// during the construction of the class.
int DeviceInfo::totalSMsOnDevice = 0;
int DeviceInfo::totalTPCsOnDevice = 0;
int DeviceInfo::maxThreadsPerSM = 0;
int DeviceInfo::SMsPerTPC = 0;

// single instance of this class which all other classes can access.
DeviceInfo *DeviceInfo::deviceProps = nullptr;

DeviceInfo *DeviceInfo::getDeviceProps() {
  if (deviceProps == nullptr) {
    // heap allocation.
    deviceProps = new DeviceInfo();
  }
  return deviceProps;
}

void DeviceInfo::initTPCMaskVector() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int SMsOnDevice = deviceProp.multiProcessorCount;
  int TPCsOnDevice = std::floor(SMsOnDevice / 2);

  for (int i = 0; i < TPCsOnDevice; i++) {
    //  based on the TPC index, the mask has to be shifted i times to the left.
    //  However, the added bits can not be zeroes because a zero indicates an
    //  enabled TPC, only one bit in the whole mask may be disabled

    uint64_t mask = ~0ULL;

    // Clear the bit at the specified index
    mask &= ~(1ULL << i);

    MaskElement element(i, true, mask);
    // push the new element to the back.
    TPCMasks.push_back(element);
  }
}

std::vector<MaskElement> DeviceInfo::getTPCMasks() { return this->TPCMasks; }

void DeviceInfo::disableTPC(int index) { this->TPCMasks.at(index).disable(); }

void DeviceInfo::enableTPC(int index) { this->TPCMasks.at(index).enable(); }

DeviceInfo::DeviceInfo() {
  this->initTPCMaskVector();
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  totalSMsOnDevice = deviceProp.multiProcessorCount;
  totalTPCsOnDevice = std::floor(totalSMsOnDevice / 2);
  // nvidia forum:
  // https://forums.developer.nvidia.com/t/why-is-max-threads-per-sm-larger-than-max-threads-per-block/277817/3
  maxThreadsPerSM = 2048;
  // more modern NVIDIA GPUs contain 2 SMs per TPC.
  SMsPerTPC = 2;
}

int DeviceInfo::getTotalSMsOnDevice() { return totalSMsOnDevice; }
int DeviceInfo::getTotalTPCsOnDevice() { return totalTPCsOnDevice; }
int DeviceInfo::getMaxThreadsPerSM() { return maxThreadsPerSM; }
int DeviceInfo::getSMsPerTPC() { return SMsPerTPC; }

void DeviceInfo::deleteDevicePropsInstance() { delete (deviceProps); }
