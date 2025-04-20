#include "busyJob.h"

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
void CUDART_CB BusyJob::busyKernelCallback(cudaStream_t stream,
                                           cudaError_t status, void *data) {

  // get the kernel launch config that has to be cleaned up and that contains
  // info to display.
  auto *kernelInfo = static_cast<KernelLaunchInfoBase *>(data);

  // push the job to the clean up queue which the scheduler will handle in its
  // own thread.
  // current time is called inside the cuda runtime thread spawned by the
  // callback => safe to call host function because it will not interfere with
  // the main thread.
  float currentTime = getCurrentTime();
  CompletionQueue::getCompletionQueue().push(
      {kernelInfo->jobPtr, currentTime, kernelInfo});
}

// callback constructor.
void BusyJob::addBusyKernelCallback(Job *job, cudaStream_t stream, float *dptr,
                                    float *hptr, size_t size, int id) {

  BusyJobLaunchInfo *kernelInfo = new BusyJobLaunchInfo(job);

  cudaStreamAddCallback(stream, busyKernelCallback, kernelInfo, 0);
}

void BusyJob::execute() {

  // Allocate memory
  if (!this->TPCMasks.empty()) {
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask((void *)(this->busyStream), mask);
  }
  maxUtilizationKernel<<<10, 10, 0, this->busyStream>>>(this->dptr, 1000000);
  // define the asynchronous memory transfer here.
  cudaMemcpyAsync(this->hptr, this->dptr, this->nrOfBytes,
                  cudaMemcpyDeviceToHost, this->busyStream);
  // this callback is called only when the kernel is finished and the memory
  // copying is finished.
  addBusyKernelCallback(this, this->busyStream, this->dptr, this->hptr,
                        sizeof(float), 1);
}

BusyJob::BusyJob(int threadsPerBlock, int threadBlocks) {
  // kernel launch configuration.
  CUDA_CHECK(cudaStreamCreate(&(this->busyStream)));
  CUDA_CHECK(cudaMalloc(&(this->dptr), sizeof(float)));
  CUDA_CHECK(cudaMalloc(&(this->timerptr), sizeof(float)));
  // allocate host memory using cuda. If done this way, copying from device to
  // host can be done asynchronously.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  this->nrOfBytes = sizeof(float);
  CUDA_CHECK(cudaHostAlloc((void **)&(this->hptr), this->nrOfBytes,
                           cudaHostAllocDefault));

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
      ceil(neededSMs / DeviceInfo::getDeviceProps()->getSMsPerTPC());
}

std::string BusyJob::getMessage() { return "busy job finished\n"; }

BusyJob::~BusyJob() {
  cudaStreamSynchronize(this->busyStream);

  cudaStreamDestroy(this->busyStream);

  // Clean up other resources
  cudaFree(this->dptr);
  cudaFree(this->timerptr);
  cudaFreeHost(this->hptr);
}
