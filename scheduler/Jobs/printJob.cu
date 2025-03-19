#include <iostream>
#include <cuda_runtime.h>






__global__ static void printMessage(int taskId, int jobId){
  printf("hello from task: %d and job: %d", taskId, jobId);
}



void executeJob(int taskId, int jobId){
  printMessage<<<1, 1>>>(taskId, jobId);

}
