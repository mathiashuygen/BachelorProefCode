#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <queue>
#include "Tasks/task.cu"
#include "Jobs/printJob.cu"
#include "Jobs/busyJob.cu"
#include "common/helpFunctions.h"


int main(){
  
  float currTime = getCurrentTime();
  std::queue<std::unique_ptr<Task>> taskQueue;
  std::queue<std::queue<std::unique_ptr<Task>>> queue;

  std::unique_ptr<Job> printer = std::make_unique<PrintJob<int, int, int>>(10, 10); 
  PrintJob<int, int, int> jobke(10, 10);

  Task task1(10, 5, 20, 100, std::move(printer), 1);
  Task task2(10, 5, 20, 100, std::move(printer), 2);

  
  while(true){

    if(task1.isJobReady()){task1.getJob()->execute(10, 10, 100);} 


    sleep(2000);

  }



  return 0;
}

