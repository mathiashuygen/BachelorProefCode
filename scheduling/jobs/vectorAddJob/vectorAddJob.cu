#include "vectorAddJob.h"
#include <cmath>

void CUDART_CB VectorAddJob::vectorAddKernelCallback(cudaStream_t stream,
                                                     cudaError_t status,
                                                     void *data) {

  // get the kernel launch config that has to be cleaned up and potentially
  // checked for correctness.
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
void VectorAddJob::addVectorAddKernelCallback(Job *job, cudaStream_t stream,
                                              float *d_A, float *d_B,
                                              float *d_C, float *A, float *B,
                                              float *C) {

  VectorAddJobLaunchInfo *kernelInfo = new VectorAddJobLaunchInfo(job);
  // register the callback for the given stream.
  cudaStreamAddCallback(stream, vectorAddKernelCallback, kernelInfo, 0);
}

void VectorAddJob::execute() {

  // set the stream's mask using libsmctrl.
  if (!this->TPCMasks.empty()) {
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask((void *)(this->kernelStream), mask);
  }

  // fill up two arrays with values.
  for (int i = 0; i < this->vectorSize; i++) {
    A[i] = i;
    B[i] = i + i;
  }

  // copy the contents of the host arrays to the device arrays in an async way
  // before the kernel is launched.
  cudaMemcpyAsync(d_A, A, nrOfElements, cudaMemcpyHostToDevice, kernelStream);
  cudaMemcpyAsync(d_B, B, nrOfElements, cudaMemcpyHostToDevice, kernelStream);

  // kernel launch.
  vectorAddKernel<<<this->threadBlocks, this->threadsPerBlock, 0,
                    kernelStream>>>(d_A, d_B, d_C, this->vectorSize);

  //  copy the result back into the host array.
  cudaMemcpyAsync(C, d_C, nrOfElements, cudaMemcpyDeviceToHost, kernelStream);
  addVectorAddKernelCallback(this, kernelStream, d_A, d_B, d_C, A, B, C);
}

VectorAddJob::VectorAddJob(int threadsPerBlock, int vectorSize) {

  this->threadsPerBlock = threadsPerBlock;
  this->vectorSize = vectorSize;

  // kernel launch config.
  cudaMalloc(&d_A, this->vectorSize * sizeof(float));
  cudaMalloc(&d_B, this->vectorSize * sizeof(float));
  cudaMalloc(&d_C, this->vectorSize * sizeof(float));

  cudaStreamCreate(&kernelStream);

  nrOfElements = this->vectorSize * sizeof(float);
  cudaHostAlloc((void **)&A, nrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&B, nrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&C, nrOfElements, cudaHostAllocDefault);

  // in this case the total amount of threads is the same as the size of the
  // vector because each thread will calculate one addition.
  int totalThreads = vectorSize;
  this->threadBlocks =
      std::ceil(totalThreads + this->threadsPerBlock - 1 / threadsPerBlock);
  int neededSMs =
      ceil((float)totalThreads /
           (float)DeviceInfo::getDeviceProps()->getMaxThreadsPerSM());

  if (neededSMs < 1) {
    this->neededTPCs = 1;
    return;
  }
  this->neededTPCs =
      ceil(neededSMs / DeviceInfo::getDeviceProps()->getSMsPerTPC());
}

std::string VectorAddJob::getMessage() { return "vector addition done\n"; }
