#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "Tasks/task.cu"
#include "Jobs/printJob.cu"
#include "Jobs/busyJob.cu"
#include "common/helpFunctions.h"
#include "schedulers/JLFP.cu"

int main(){
  
  
  std::vector<Task*> tasks;

  std::unique_ptr<Job> printer = std::make_unique<PrintJob<int, int, int>>(10, 10); 
  std::unique_ptr<Job> busy = std::make_unique<BusyJob<int, int, int>>(10, 10);
  
  

  Task task1(10, 5, 20, 100, std::move(printer), 1);
  Task task2(10, 5, 20, 100, std::move(busy), 2);

  tasks.push_back(&task1);
  tasks.push_back(&task2);

  JLFP scheduler1;
  
  while(true){

    for(Task* task : tasks){
      if(task->isJobReady()){
        scheduler1.addJob(task->releaseJob());
      }
    }
    scheduler1.displayQueuePriorities();
    sleep(2000);

  }



  return 0;
}

