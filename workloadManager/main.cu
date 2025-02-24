#include "thrust/device_vector.h"
#include <iostream>
#include <iomanip>
#include "cuda_runtime.h"
#include "computeIntensive.h"
#include <chrono>

__host__ void prepareArguments(int* threadsPerBlock, int* blocksPerSM, int* totalBlocks, int* arraySize, int totalSMs){
    *threadsPerBlock = 256;
    *blocksPerSM = 32;
    *totalBlocks = *blocksPerSM * totalSMs;
    *arraySize = *totalBlocks * *threadsPerBlock;

    return;
}



__host__ void launchComputeIntensiveTask(){
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int totalSMs = deviceProp.multiProcessorCount;
   


  int threadsPerBlock, blocksPerSM, totalBlocks, arraySize;

  prepareArguments(&threadsPerBlock, &blocksPerSM, &totalBlocks, &arraySize, totalSMs);

  thrust::device_vector<float>workArray(arraySize);

  auto start = std::chrono::high_resolution_clock::now();

  maxUtilizationKernel<<<totalBlocks, threadsPerBlock>>>(workArray, arraySize, 1000);
  

  cudaDeviceSynchronize();

 
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> total = end - start;

  std::cout << std::fixed << std::setprecision(0) << "Finished execution, the kernel took " << total.count() << " ms" << std::endl;
  return;
}

int main(int argc, char *argv[])
{
    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int threadsPerBlock, blocksPerSM, totalBlock;
    

    while(true){
      
      std::string option;

      std::cout <<"choose type of task: \n"; 
      std::cout <<"\t-(1)type 1: compute intensive\n";
      std::cout <<"\t-(2)type 2: memory intensive\n";
      std::cout <<"\t-(3)exit\n";

      std::getline(std::cin, option);
      
      if(option == "1"){
        std::cout<<"chosen compute intensive task\n";
        launchComputeIntensiveTask();
      }
      else if(option == "2"){
        std::cout<<"chosen memory intensive task\n";
      }
      else if(option == "3"){
        std::cout<<"chosen to exit\n";
        return 0;
      }
      else{
        std::cout<<"not a valid option chose, choose between 1-2-3\n";
      }

    }    


    cudaDeviceSynchronize();
    cudaDeviceReset();
    
    
    return 0;
}
