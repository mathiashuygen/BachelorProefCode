#include <any>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include "kernels/printKernel.cu"
#include "job.h"

template<typename... FuncArgs>
class PrintJob: public Job{
  private:
    int minimumCores, maximumCores;

    struct printKernelLaunchInformation {
      cudaStream_t kernelStream;
      float* devicePtr;    // Device memory pointer
      float* hostPtr;      // Host memory pointer
      size_t size;        // Size of data to copy in bytes
      int taskId;         //id of  the task invoking jobs.
    

      printKernelLaunchInformation(cudaStream_t stream, float* dptr, float* hptr, size_t sz, int id)
        : kernelStream(stream), devicePtr(dptr), hostPtr(hptr), size(sz), taskId(id){}

    };



    //callback that is envoked at the end of each kernel execution.
    static void CUDART_CB printKernelCallback(cudaStream_t stream, cudaError_t status, void *data){

      //get the kernel launch config that has to be cleaned up and that contains info to display.
      printKernelLaunchInformation* kernelInfo = static_cast<printKernelLaunchInformation*>(data);
  
      //copy the result from device to host.
      cudaMemcpy(kernelInfo->hostPtr, kernelInfo->devicePtr, kernelInfo->size, cudaMemcpyDeviceToHost);

      std::cout<<"job from task "<<kernelInfo->taskId<<" took "<<*(kernelInfo->hostPtr)<<"s\n";
  
      //free the dynamically allocated memory and the stream.
      free(kernelInfo->hostPtr);
      cudaFree(kernelInfo->devicePtr);
      cudaStreamDestroy(stream);

    } 


    //callback constructor.
    static void addPrintKernelCallback(cudaStream_t stream, float* dptr, float* hptr, size_t size, int id){

      printKernelLaunchInformation* kernelInfo = new printKernelLaunchInformation(stream, dptr, hptr, size, id);

      cudaStreamAddCallback(stream, printKernelCallback, kernelInfo, 0);
    }

  public:
    
    //job execute function.
   static void executeJob(int taskId, int jobId, int loopDuration){
      std::cout<<"execute\n";  
      cudaStream_t kernel_stream;
      cudaStreamCreate(&kernel_stream);
      float *d_output;
      float *h_output = (float*)std::malloc(sizeof(float));
      cudaMalloc(&d_output, sizeof(float));

  

    printMessage<<<1, 1, 0, kernel_stream>>>(taskId, jobId, loopDuration, d_output);
    addPrintKernelCallback(kernel_stream, d_output, h_output, sizeof(float), taskId);

    return;

    }


    PrintJob(int minimumCores, int maximumCores): minimumCores(minimumCores), maximumCores(maximumCores)
    {} 




    void launchJob() override{
      auto& args_tuple = std::any_cast<std::tuple<FuncArgs...>&>(args);
      std::apply(executeJob, args_tuple);
    }

};

