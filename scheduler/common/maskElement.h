
#include <cstdint>

class MaskElement {

private:
  int index;
  bool free;
  uint64_t TPCMask;

public:
  MaskElement(int index, bool free, uint64_t mask)
      : index(index), free(free), TPCMask(mask) {}

  int getIndex();
  bool isFree();
  uint64_t getMask();
  void disable();
  void enable();
};
