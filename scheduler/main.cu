#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "Tasks/task.cu"
#include "Jobs/printJob.cu"
#include "Jobs/busyJob.cu"
#include "Jobs/jobFactory.h"
#include "common/helpFunctions.h"
#include "schedulers/JLFP.cu"

int main(){
  
  
  std::vector<Task*> tasks;

  auto printJobFactory = TemplatedJobFactoryHelper<PrintJob, int, int, int>::create(10, 10); 
  auto busyJobFactory = TemplatedJobFactoryHelper<BusyJob, int, int, int>::create(10, 10);

  Task task1(10, 5, 20, 100, std::move(printJobFactory), 1);
  Task task2(10, 5, 20, 100, std::move(busyJobFactory), 2);

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
    scheduler1.dispatch();
    sleep(2000);

  }



  return 0;
}

