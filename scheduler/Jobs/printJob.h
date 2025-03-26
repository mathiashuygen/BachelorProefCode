#ifndef PRINT_JOB_H
#define PRINT_JOB_H



#include "kernels/printKernel.h"
#include <iostream>
#include "job.h"
#include <cuda_runtime.h>


class PrintJob: public Job{

  private:

    struct printKernelLaunchInformation {
      cudaStream_t kernelStream; //stream in which the kernel is launched. 
      float*       devicePtr;    // Device memory pointer
      float*       hostPtr;      // Host memory pointer
      size_t       size;         // Size of data to copy in bytes
      int          taskId;       //id of  the task invoking jobs.
    

      printKernelLaunchInformation(cudaStream_t stream, float* dptr, float* hptr, size_t sz, int id)
        : kernelStream(stream), devicePtr(dptr), hostPtr(hptr), size(sz), taskId(id){
        }

    };
    static void CUDART_CB printKernelCallback(cudaStream_t stream, cudaError_t status, void *data);
    static void addPrintKernelCallback(cudaStream_t stream, float* dptr, float* hptr, size_t size, int id);
    

  public:
    virtual void execute() override;

    PrintJob(int minimumTPCs, int maximumTPCs);  




};




#endif
