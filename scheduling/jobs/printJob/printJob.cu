
#include "printJob.h"

#define CUDA_CHECK(err) __cudaSafeCall(err, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    std::cerr << "CUDA error in " << file << ":" << line << " : "
              << cudaGetErrorString(err) << " (" << err << ")" << std::endl;
    // Consider more graceful error handling than exit depending on your app
    exit(EXIT_FAILURE);
  }
}
// callback that is envoked at the end of each kernel execution.
void CUDART_CB PrintJob::printKernelCallback(cudaStream_t stream,
                                             cudaError_t status, void *data) {

  auto *kernelInfo = static_cast<KernelLaunchInfoBase *>(data);

  // Just push to completion queue with the cleanup info
  float currentTime = getCurrentTime();
  CompletionQueue::getCompletionQueue().push(
      {kernelInfo->jobPtr, currentTime, kernelInfo});
}

// callback constructor.
void PrintJob::addPrintKernelCallback(Job *job, cudaStream_t stream,
                                      float *dptr, float *hptr, size_t size,
                                      int id) {

  PrintJobLaunchInfo *kernelInfo =
      new PrintJobLaunchInfo(job, stream, dptr, hptr, size, id);
  cudaStreamAddCallback(stream, printKernelCallback, kernelInfo, 0);
}

// job execute function.
void PrintJob::execute() {
  // Allocate memory
  if (!this->TPCMasks.empty()) {
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask(this->jobStream, mask);
  }
  printMessage<<<1, 1, 0, this->jobStream>>>(1, 1, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpyAsync(this->hostPtr, this->dptr, this->nbrOfBytes,
                             cudaMemcpyDeviceToHost, this->jobStream));

  addPrintKernelCallback(this, this->jobStream, this->dptr, this->hostPtr,
                         sizeof(float), 1);
  CUDA_CHECK(cudaGetLastError());
  return;
}

PrintJob::PrintJob(int threadsPerBlock, int threadBlocks) {
  this->threadsPerBlock = threadsPerBlock;
  this->threadBlocks = threadBlocks;

  int totalThreads = threadsPerBlock * threadBlocks;
  int neededSMs =
      totalThreads / DeviceInfo::getDeviceProps()->getMaxThreadsPerSM();

  // prepare the kernel launch.
  CUDA_CHECK(cudaStreamCreate(&(this->jobStream)));
  CUDA_CHECK(cudaMalloc(&(this->dptr), sizeof(float)));
  // allocate host memory using cuda. If done this way, copying from device to
  // host can be done asynchronously.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  this->nbrOfBytes = sizeof(float);
  CUDA_CHECK(cudaHostAlloc((void **)&(this->hostPtr), nbrOfBytes,
                           cudaHostAllocDefault));
  if (neededSMs < 1) {
    this->neededTPCs = 1;
    return;
  }
  this->neededTPCs =
      ceil(neededSMs / DeviceInfo::getDeviceProps()->getSMsPerTPC());
}

std::string PrintJob::getMessage() { return "print job finished\n"; }

PrintJob::~PrintJob() {
  cudaStreamSynchronize(jobStream);

  cudaStreamDestroy(jobStream);

  // Clean up other resources
  cudaFree(dptr);
  cudaFreeHost(hostPtr);
}
