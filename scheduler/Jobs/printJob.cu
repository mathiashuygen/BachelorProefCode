#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>



__global__ static void printMessage(int taskId, int jobId, int loopDuration, float* timing){
  
  float startTime = clock();
  for(int i = 0; i < loopDuration; i++){
    float y = 0;
    y = sinf(10.2) + cosf(3.1);
  }
  float endTime = clock();

  *timing = endTime - startTime;

}


struct kernelLaunchInformation {
    cudaStream_t kernelStream;
    float* devicePtr;    // Device memory pointer
    float* hostPtr;      // Host memory pointer
    size_t size;        // Size of data to copy in bytes
    int taskId;         //id of  the task invoking jobs.
    

    kernelLaunchInformation(cudaStream_t stream, float* dptr, float* hptr, size_t sz, int id)
        : kernelStream(stream), devicePtr(dptr), hostPtr(hptr), size(sz), taskId(id){}

};



//callback that is envoked at the end of each kernel execution.
void CUDART_CB printKernelCallback(cudaStream_t stream, cudaError_t status, void *data){

  //get the kernel launch config that has to be cleaned up and that contains info to display.
  kernelLaunchInformation* kernelInfo = static_cast<kernelLaunchInformation*>(data);
  
  //copy the result from device to host.
  cudaMemcpy(kernelInfo->hostPtr, kernelInfo->devicePtr, kernelInfo->size, cudaMemcpyDeviceToHost);

  std::cout<<"job from task "<<kernelInfo->taskId<<" took "<<*(kernelInfo->hostPtr)<<"s\n";
  
  //free the dynamically allocated memory and the stream.
  free(kernelInfo->hostPtr);
  cudaFree(kernelInfo->devicePtr);
  cudaStreamDestroy(stream);

} 


//callback constructor.
void addKernelCallback(cudaStream_t stream, float* dptr, float* hptr, size_t size, int id){

  kernelLaunchInformation* kernelInfo = new kernelLaunchInformation(stream, dptr, hptr, size, id);

  cudaStreamAddCallback(stream, printKernelCallback, kernelInfo, 0);

}


//job definition that goes with a task.
void printJobExec(int taskId, int jobId, int loopDuration){
  
  cudaStream_t kernel_stream;
  cudaStreamCreate(&kernel_stream);
  float *d_output;
  float *h_output = (float*)std::malloc(sizeof(float));
  cudaMalloc(&d_output, sizeof(float));

  

  printMessage<<<1, 1, 0, kernel_stream>>>(taskId, jobId, loopDuration, d_output);
  addKernelCallback(kernel_stream, d_output, h_output, sizeof(float), taskId);

  return;

}

