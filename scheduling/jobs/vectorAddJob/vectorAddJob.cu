#include "vectorAddJob.h"
#include <cmath>

void CUDART_CB VectorAddJob::vectorAddKernelCallback(cudaStream_t stream,
                                                     cudaError_t status,
                                                     void *data) {

  // get the kernel launch config that has to be cleaned up and potentially
  // checked for correctness.
  VectorAddJob::vectorAddKernelLaunchInformation *kernelInfo =
      static_cast<VectorAddJob::vectorAddKernelLaunchInformation *>(data);

  // free the dynamically allocated memory and the stream.
  cudaFreeHost(kernelInfo->A);
  cudaFreeHost(kernelInfo->B);
  cudaFreeHost(kernelInfo->C);
  cudaFree(kernelInfo->d_A);
  cudaFree(kernelInfo->d_B);
  cudaFree(kernelInfo->d_C);

  cudaStreamDestroy(stream);

  // push the job to the clean up queue which the scheduler will handle in its
  // own thread.
  // current time is called inside the cuda runtime thread spawned by the
  // callback => safe to call host function because it will not interfere with
  // the main thread.
  float currentTime = getCurrentTime();
  CompletionQueue::getCompletionQueue().push({kernelInfo->jobPtr, currentTime});

  // destroy the struct.
  delete (kernelInfo);
}

// callback constructor.
void VectorAddJob::addVectorAddKernelCallback(Job *job, cudaStream_t stream,
                                              float *d_A, float *d_B,
                                              float *d_C, float *A, float *B,
                                              float *C) {

  VectorAddJob::vectorAddKernelLaunchInformation *kernelInfo =
      new VectorAddJob::vectorAddKernelLaunchInformation(job, stream, d_A, d_B,
                                                         d_C, A, B, C);
  // register the callback for the given stream.
  cudaStreamAddCallback(stream, vectorAddKernelCallback, kernelInfo, 0);
}

void VectorAddJob::execute() {

  // device vectors.
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, this->vectorSize * sizeof(float));
  cudaMalloc(&d_B, this->vectorSize * sizeof(float));
  cudaMalloc(&d_C, this->vectorSize * sizeof(float));

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  // set the stream's mask using libsmctrl.
  if (!this->TPCMasks.empty()) {
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask(kernel_stream, mask);
  }
  // allocate host memory using cuda. If done this way, copying from device to
  // host can be done asynchronously.
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gb65da58f444e7230d3322b6126bb4902
  float *A = nullptr;
  float *B = nullptr;
  float *C = nullptr;

  size_t nbrOfElements = this->vectorSize * sizeof(float);
  cudaHostAlloc((void **)&A, nbrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&B, nbrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&C, nbrOfElements, cudaHostAllocDefault);

  // fill up two arrays with values.
  for (int i = 0; i < this->vectorSize; i++) {
    A[i] = i;
    B[i] = i + i;
  }

  // copy the contents of the host arrays to the device arrays in an async way
  // before the kernel is launched.
  cudaMemcpyAsync(d_A, A, nbrOfElements, cudaMemcpyHostToDevice, kernel_stream);
  cudaMemcpyAsync(d_B, B, nbrOfElements, cudaMemcpyHostToDevice, kernel_stream);

  // kernel launch.
  vectorAddKernel<<<this->threadBlocks, this->threadsPerBlock, 0,
                    kernel_stream>>>(d_A, d_B, d_C, this->vectorSize);

  //  copy the result back into the host array.
  cudaMemcpyAsync(C, d_C, nbrOfElements, cudaMemcpyDeviceToHost, kernel_stream);
  addVectorAddKernelCallback(this, kernel_stream, d_A, d_B, d_C, A, B, C);
}

VectorAddJob::VectorAddJob(int threadsPerBlock, int vectorSize) {
  this->threadsPerBlock = threadsPerBlock;
  this->vectorSize = vectorSize;
  std::cout << this->vectorSize;

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
  std::cout << "needed tpcs = " << neededTPCs << "\n";
}

std::string VectorAddJob::getMessage() { return "vector addition done\n"; }
