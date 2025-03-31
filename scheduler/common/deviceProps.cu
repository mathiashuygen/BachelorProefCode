#include "deviceProps.h"
#include <cmath>
#include <cstdint>

void DeviceInfo::initTPCMaskVector() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int SMsOnDevice = deviceProp.multiProcessorCount;
  int TPCsOnDevice = std::floor(SMsOnDevice / 2);

  for (int i = 0; i < TPCsOnDevice; i++) {
    uint64_t baseMask = 0xFFFFFFFF;
    // intermediate mask is a mask that reserves one TPC, based on the TPC
    // index, the mask has to be shifted to the left. However, the added bits
    // can not be zeroes because a zero indicates an enabled TPC, only one bit
    // in the whole mask may be disabled.
    uint64_t intermediateMAsk = baseMask << 1;
    // now get the disbaled bit at the right index and force every added bit to
    // be a one.
    uint64_t fullMask = (intermediateMAsk << i) | ((1 << i) - 1);
    // construct a new element for the vector.
    maskElement element(true, fullMask, i);
    // push the new element to the back.
    TPCMasks.push_back(element);
  }
}

std::vector<DeviceInfo::maskElement> DeviceInfo::getTPCMasks() {
  return this->TPCMasks;
}

void DeviceInfo::disableTPC(int index) {
  this->TPCMasks.at(index).free = false;
}

void DeviceInfo::enableTPC(int index) { this->TPCMasks.at(index).free = true; }

DeviceInfo::DeviceInfo() { this->initTPCMaskVector(); }
