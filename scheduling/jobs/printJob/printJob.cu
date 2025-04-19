
#include "printJob.h"

// callback that is envoked at the end of each kernel execution.
void CUDART_CB PrintJob::printKernelCallback(cudaStream_t stream,
                                             cudaError_t status, void *data) {

  // get the kernel launch config that has to be cleaned up and that contains
  // info to display.
  printKernelLaunchInformation *kernelInfo =
      static_cast<printKernelLaunchInformation *>(data);

  // free the dynamically allocated memory and the stream.
  cudaFreeHost(kernelInfo->hostPtr);
  cudaFree(kernelInfo->devicePtr);
  cudaStreamDestroy(stream);

  // push completed job to async clean up queue.
  float currentTime = getCurrentTime();
  CompletionQueue::getCompletionQueue().push({kernelInfo->jobPtr, currentTime});

  delete (kernelInfo);
}

// callback constructor.
void PrintJob::addPrintKernelCallback(Job *job, cudaStream_t stream,
                                      float *dptr, float *hptr, size_t size,
                                      int id) {

  printKernelLaunchInformation *kernelInfo =
      new printKernelLaunchInformation(job, stream, dptr, hptr, size, id);
  cudaStreamAddCallback(stream, printKernelCallback, kernelInfo, 0);
}

// job execute function.
void PrintJob::execute() {
  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  float *d_output;
  cudaMalloc(&d_output, sizeof(float));
  // allocate host memory using cuda. If done this way, copying from device to
  // host can be done asynchronously.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  float *h_output = nullptr;
  size_t nbrOfBytes = sizeof(float);
  cudaHostAlloc((void **)&h_output, nbrOfBytes, cudaHostAllocDefault);

  printMessage<<<1, 1, 0, kernel_stream>>>(1, 1, 1, d_output);
  cudaMemcpyAsync(h_output, d_output, nbrOfBytes, cudaMemcpyHostToDevice,
                  kernel_stream);

  addPrintKernelCallback(this, kernel_stream, d_output, h_output, sizeof(float),
                         1);

  return;
}

PrintJob::PrintJob(int threadsPerBlock, int threadBlocks) {
  this->threadsPerBlock = threadsPerBlock;
  this->threadBlocks = threadBlocks;

  int totalThreads = threadsPerBlock * threadBlocks;
  int neededSMs =
      totalThreads / DeviceInfo::getDeviceProps()->getMaxThreadsPerSM();

  if (neededSMs < 1) {
    this->neededTPCs = 1;
    return;
  }
  this->neededTPCs =
      int(ceil(neededSMs / DeviceInfo::getDeviceProps()->getSMsPerTPC()));
}

std::string PrintJob::getMessage() { return "print job finished\n"; }
