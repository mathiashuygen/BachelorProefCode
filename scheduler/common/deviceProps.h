#include <cmath>
#include <cstdint>
#include <sys/types.h>
#include <vector>

class DeviceInfo {
public:
  struct maskElement {
    bool free;
    uint64_t TPCMask;
    int index;

    maskElement(bool free, u_int64_t mask, int index)
        : free(free), TPCMask(mask), index(index) {}
  };

  std::vector<maskElement> TPCMasks;

  void initTPCMaskVector();

  std::vector<maskElement> getTPCMasks();

  void disableTPC(int index);
  void enableTPC(int index);

  DeviceInfo();
};
