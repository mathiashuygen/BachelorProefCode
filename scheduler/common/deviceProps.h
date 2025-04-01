#include "maskElement.h"
#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <vector>
class DeviceInfo {

private:
  static DeviceInfo *deviceProps;
  DeviceInfo();

public:
  std::vector<MaskElement> TPCMasks;

  void initTPCMaskVector();

  std::vector<MaskElement> getTPCMasks();

  void disableTPC(int index);
  void enableTPC(int index);
  static DeviceInfo *getDeviceProps();
  // clean-up code.
  void deleteDevicePropsInstance();
};
