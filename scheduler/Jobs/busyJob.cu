#include <iostream>
#include <cuda_runtime.h>
#include "kernels/busyKernel.cu"
#include "job.h"



template<typename... FuncArgs>
class BusyJob: public Job{
  private:
    int minimumTPCs, maximumTPCs;
    float releaseTime, maximalExecutionTime, absoluteDeadline;





    struct busyKernelLaunchInformation {
      cudaStream_t kernelStream; //stream in which the kernel is launched. 
      float*       devicePtr;    // Device memory pointer
      float*       timerDptr;    //device ptr to float that holds the total execution time of a kernel.
      float*       hostPtr;      // Host memory pointer
      size_t       size;         // Size of data to copy in bytes
      int          taskId;       //id of  the task invoking jobs.
    

      busyKernelLaunchInformation(cudaStream_t stream, float* dptr, float* timeDptr, float* hptr, size_t sz, int id)
        : kernelStream(stream), devicePtr(dptr), timerDptr(), hostPtr(hptr), size(sz), taskId(id){}

    };



    //callback that is envoked at the end of each kernel execution.
    static void CUDART_CB busyKernelCallback(cudaStream_t stream, cudaError_t status, void *data){

      //get the kernel launch config that has to be cleaned up and that contains info to display.
      busyKernelLaunchInformation* kernelInfo = static_cast<busyKernelLaunchInformation*>(data);
  
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
    static void addBusyKernelCallback(cudaStream_t stream, float* dptr, float* timerDptr, float* hptr, size_t size, int id){

      busyKernelLaunchInformation* kernelInfo = new busyKernelLaunchInformation(stream, dptr, timerDptr, hptr, size, id);

      cudaStreamAddCallback(stream, busyKernelCallback, kernelInfo, 0);

    }



  public:

    //job definition that goes with a task.
    static void executeJob(int taskId, int jobId, int loopDuration){
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
  
  

      maxUtilizationKernel<<<1, 1, 0, kernel_stream>>>(d_output, d_timer, loopDuration);
      addBusyKernelCallback(kernel_stream, d_output, d_timer, h_output, sizeof(float), taskId);

      return;

    } 



    BusyJob(int minimumTPCs, int maximumTPCs): minimumTPCs(minimumTPCs), maximumTPCs(maximumTPCs)
    {} 




    void launchJob() override{
      auto& args_tuple = std::any_cast<std::tuple<FuncArgs...>&>(args);
      std::apply(executeJob, args_tuple);
    }

};




