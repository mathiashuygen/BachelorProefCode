#include "matrixMultiplicationJob.h"

void CUDART_CB MatrixMultiplicationJob::matrixMulKernelCallback(
    cudaStream_t stream, cudaError_t status, void *data) {

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

void MatrixMultiplicationJob::addMatrixMulKernelCallback(Job *job,
                                                         cudaStream_t stream) {
  MatrixMulJobLaunchInfo *kernelInfo = new MatrixMulJobLaunchInfo(job);
  // register the callback for the given stream.
  cudaStreamAddCallback(stream, matrixMulKernelCallback, kernelInfo, 0);
}

void MatrixMultiplicationJob::execute() {
  // get a random absolute deadline.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> realDist(1.0, 100.0);

  // set the stream's mask using libsmctrl.
  // TODO: this is pure code duplication => create method inside base job class
  // to do this.
  if (!this->TPCMasks.empty()) {
    uint64_t mask = this->combineMasks();
    libsmctrl_set_stream_mask((void *)(this->kernelStream), mask);
  }

  // fill up two arrays with values.
  for (int i = 0; i < this->nrOfElements; i++) {

    A[i] = realDist(gen);
    B[i] = realDist(gen);
  }

  // copy the contents of the host arrays to the device arrays in an async way
  // before the kernel is launched.
  cudaMemcpyAsync(d_A, A, nrOfElements, cudaMemcpyHostToDevice, kernelStream);
  cudaMemcpyAsync(d_B, B, nrOfElements, cudaMemcpyHostToDevice, kernelStream);

  // kernel launch.
  matrixMul<<<this->blocks, this->threadsPerBlock, 0, kernelStream>>>(
      d_A, d_B, d_C, this->matrixDim);

  //  copy the result back into the host array.
  cudaMemcpyAsync(C, d_C, nrOfElements, cudaMemcpyDeviceToHost, kernelStream);
  MatrixMultiplicationJob::addMatrixMulKernelCallback(this, this->kernelStream);
}

MatrixMultiplicationJob::MatrixMultiplicationJob(int threadsPerBlock,
                                                 int matrixDim) {
  this->threadsPerBlock = threadsPerBlock;
  this->matrixDim = matrixDim;

  this->nrOfElements = this->matrixDim * this->matrixDim * sizeof(float);
  // kernel launch config.
  // matrices => 2D arrays => matrixDim * matrixDim.
  cudaMalloc(&d_A, this->nrOfElements);
  cudaMalloc(&d_B, this->nrOfElements);
  cudaMalloc(&d_C, this->nrOfElements);

  cudaStreamCreate(&kernelStream);

  cudaHostAlloc((void **)&A, this->nrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&B, this->nrOfElements, cudaHostAllocDefault);
  cudaHostAlloc((void **)&C, this->nrOfElements, cudaHostAllocDefault);

  // in this case the total amount of threads is the same as the size of the
  // matrix because each thread will calculate one row and column
  // multiplication..
  int totalThreads = this->nrOfElements;
  this->blocks = dim3(
      (this->matrixDim + this->threadsPerBlock - 1 / this->threadsPerBlock),
      (this->matrixDim + this->threadsPerBlock - 1 / this->threadsPerBlock));
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

std::string MatrixMultiplicationJob::getMessage() {
  return "matrix mul finished\n";
}

MatrixMultiplicationJob::~MatrixMultiplicationJob() {
  cudaStreamSynchronize(this->kernelStream);

  cudaStreamDestroy(this->kernelStream);

  // Clean up other resources
  cudaFree(this->d_A);
  cudaFree(this->d_B);
  cudaFree(this->d_C);
  cudaFreeHost(this->A);
  cudaFreeHost(this->B);
  cudaFreeHost(this->C);
}
