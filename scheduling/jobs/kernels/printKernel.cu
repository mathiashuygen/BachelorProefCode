#include "printKernel.h"

__global__ void printMessage(int taskId, int jobId, int loopDuration) {

  for (int i = 0; i < loopDuration; i++) {
    float y = 0;
    y = sinf(10.2) + cosf(3.1);
  }
}
