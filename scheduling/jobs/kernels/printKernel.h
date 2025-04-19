
#ifndef PRINT_KERNEL_H
#define PRINT_KERNEL_H

#include <cuda_runtime.h>

/**
 * @param taskId
 * id of the task launching the job.
 * @param jobId
 * id of the job executing the kernel.
 * @param loopDuration.
 * int that determines how many iterations the for loop contains.
 * @param timing
 * float that holds the duration of the kernel execution.
 *
 * kernel that spins for a while. Does not print anything but is used for
 * printing the duration of a kernel.
 */
__global__ void printMessage(int taskId, int jobId, int loopDuration);

#endif
