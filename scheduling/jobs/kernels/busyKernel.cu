#include "busyKernel.h"

// Kernel that performs a simple computation, in this case only one thread will
// execute the kernel (testng sake).
__global__ void maxUtilizationKernel(float *output, int loopDuration) {

  // Some computations to keep the SM busy
  float value = 0.0f;
  for (int i = 0; i < loopDuration; i++) {
    value += sinf(10 * 0.1f + i) * cosf(10 * 0.1f);
    *output = value;
  }
}
