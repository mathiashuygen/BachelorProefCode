
#include "printJob.h"


  

//callback that is envoked at the end of each kernel execution.
void CUDART_CB PrintJob::printKernelCallback(cudaStream_t stream, cudaError_t status, void *data){

  //get the kernel launch config that has to be cleaned up and that contains info to display.
  printKernelLaunchInformation* kernelInfo = static_cast<printKernelLaunchInformation*>(data);
  
  //copy the result from device to host.
  cudaMemcpy(kernelInfo->hostPtr, kernelInfo->devicePtr, kernelInfo->size, cudaMemcpyDeviceToHost);

  //std::cout<<"print job from task "<<kernelInfo->taskId<<" took "<<*(kernelInfo->hostPtr)<<"s\n";
  
  //free the dynamically allocated memory and the stream.
  free(kernelInfo->hostPtr);
  cudaFree(kernelInfo->devicePtr);
  cudaStreamDestroy(stream);

  Job::notifyJobCompletion(kernelInfo->jobPtr);

} 


//callback constructor.
void PrintJob::addPrintKernelCallback(Job* job, cudaStream_t stream, float* dptr, float* hptr, size_t size, int id){

  printKernelLaunchInformation* kernelInfo = new printKernelLaunchInformation(job, stream, dptr, hptr, size, id);
      cudaStreamAddCallback(stream, printKernelCallback, kernelInfo, 0);
}


//job execute function.
void PrintJob::execute(){
  std::cout<<"execute\n";  
  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  float *d_output;
  float *h_output = (float*)std::malloc(sizeof(float));
  cudaMalloc(&d_output, sizeof(float));



  printMessage<<<1, 1, 0, kernel_stream>>>(1, 1, 100, d_output);
  addPrintKernelCallback(this, kernel_stream, d_output, h_output, sizeof(float), 1);

  return;

}


PrintJob::PrintJob(int minimumTPCs, int maximumTPCs){
  this->minimumTPCs = minimumTPCs;
  this->maximumTPCs = maximumTPCs;

} 




