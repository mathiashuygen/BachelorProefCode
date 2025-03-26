   
#include "busyJob.h"
    //callback that is envoked at the end of each kernel execution.
void CUDART_CB BusyJob::busyKernelCallback(cudaStream_t stream, cudaError_t status, void *data){

  //get the kernel launch config that has to be cleaned up and that contains info to display.
  BusyJob::busyKernelLaunchInformation* kernelInfo = static_cast<BusyJob::busyKernelLaunchInformation*>(data);
  
  //copy the result from device to host.
  cudaMemcpy(kernelInfo->hostPtr, kernelInfo->devicePtr, kernelInfo->size, cudaMemcpyDeviceToHost);

  std::cout<<"busy job from task "<<kernelInfo->taskId<<" took "<<*(kernelInfo->hostPtr)<<"s\n";
  
  //free the dynamically allocated memory and the stream.
  free(kernelInfo->hostPtr);
  cudaFree(kernelInfo->devicePtr);
  cudaFree(kernelInfo->timerDptr);
  cudaStreamDestroy(stream);

}   


//callback constructor.
void BusyJob::addBusyKernelCallback(cudaStream_t stream, float* dptr, float* timerDptr, float* hptr, size_t size, int id){

  BusyJob::busyKernelLaunchInformation* kernelInfo = new BusyJob::busyKernelLaunchInformation(stream, dptr, timerDptr, hptr, size, id);

  cudaStreamAddCallback(stream, busyKernelCallback, kernelInfo, 0);

}



  

//job definition that goes with a task.
void BusyJob::execute(){
  // Get device cudaGetDeviceProperties
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
    

    
  // Allocate memory
  float *d_output, *d_timer;
  cudaMalloc(&d_output, sizeof(float));
  cudaMalloc(&d_timer, sizeof(float));

  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);

  float *h_output = (float*)std::malloc(sizeof(float));
  
  

  maxUtilizationKernel<<<1, 1, 0, kernel_stream>>>(d_output, d_timer, 1000);
  addBusyKernelCallback(kernel_stream, d_output, d_timer, h_output, sizeof(float), 1);

  return;

} 



BusyJob::BusyJob(int minimumTPCs, int maximumTPCs){
  this->maximumTPCs = maximumTPCs;
  this->minimumTPCs = minimumTPCs;
} 







