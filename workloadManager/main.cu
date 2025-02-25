#include "thrust/device_vector.h"
#include <iostream>
#include <iomanip>
#include "cuda_runtime.h"
#include "computeIntensive.h"
#include "memoryIntensive.h"
#include <chrono>

__host__ void prepareArguments(int* threadsPerBlock, int* blocksPerSM, int* totalBlocks, long* arraySize, int totalSMs){

    *threadsPerBlock = 256;
    *blocksPerSM = 32;
    *totalBlocks = *blocksPerSM * totalSMs;
    *arraySize = 200000000;

    return;
}



__host__ void launchComputeIntensiveTask(){
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int totalSMs = deviceProp.multiProcessorCount;
   


  int threadsPerBlock, blocksPerSM, totalBlocks;
  long arraySize;

  prepareArguments(&threadsPerBlock, &blocksPerSM, &totalBlocks, &arraySize, totalSMs);

  thrust::device_vector<float>workArray(arraySize);
  float *dptr = thrust::raw_pointer_cast(&workArray[0]);

  auto start = std::chrono::high_resolution_clock::now();

  maxUtilizationKernel<<<totalBlocks, threadsPerBlock>>>(dptr, arraySize, 1000);
  

  cudaDeviceSynchronize();

 
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> total = end - start;

  std::cout << std::fixed << std::setprecision(0) << "Finished compute intensive execution, the kernel took " << total.count() << " ms" << std::endl;
  return;
}






__host__ void launchMemoryIntensiveTask(){
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int totalSMs = deviceProp.multiProcessorCount;
   


  int threadsPerBlock, blocksPerSM, totalBlocks;
  long arraySize;

  prepareArguments(&threadsPerBlock, &blocksPerSM, &totalBlocks, &arraySize, totalSMs);

  thrust::device_vector<float> A(arraySize);
  thrust::device_vector<float> B(arraySize);
  thrust::device_vector<float> C(arraySize);
  thrust::device_vector<float> D(arraySize);
  thrust::device_vector<float> E(arraySize);
  thrust::device_vector<float> F(arraySize);
  
  float *aptr = thrust::raw_pointer_cast(&A[0]);
  float *bptr = thrust::raw_pointer_cast(&B[0]);
  float *cptr = thrust::raw_pointer_cast(&C[0]);
  float *dptr = thrust::raw_pointer_cast(&D[0]);

  auto start = std::chrono::high_resolution_clock::now();

  highMemoryUsage<<<totalBlocks, threadsPerBlock>>>(aptr, bptr, cptr, dptr, arraySize);
  

  cudaDeviceSynchronize();

 
  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> total = end - start;

  std::cout << std::fixed << std::setprecision(0) << "Finished memory intensive execution, the kernel took " << total.count() << " ms" << std::endl;
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
        launchMemoryIntensiveTask();
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
