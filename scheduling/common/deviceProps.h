#include "maskElement.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sys/types.h>
#include <vector>
class DeviceInfo {

private:
  static DeviceInfo *deviceProps;
  DeviceInfo();
  static int totalSMsOnDevice;
  static int totalTPCsOnDevice;
  static int maxThreadsPerSM;
  static int SMsPerTPC;

public:
  std::vector<MaskElement> TPCMasks;

  void initTPCMaskVector();

  std::vector<MaskElement> getTPCMasks();

  static int getTotalSMsOnDevice();
  static int getTotalTPCsOnDevice();
  static int getMaxThreadsPerSM();
  static int getSMsPerTPC();

  void disableTPC(int index);
  void enableTPC(int index);
  static DeviceInfo *getDeviceProps();
  // clean-up code.
  void deleteDevicePropsInstance();
};
