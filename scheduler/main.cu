#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "Tasks/task.h"
#include "Jobs/printJob.h"
#include "Jobs/busyJob.h"
#include "Jobs/jobFactory.h"
#include "common/helpFunctions.h"
#include "schedulers/JLFP.h"

int main(){
  
  
  std::vector<Task*> tasks;

  auto printJobFactory = JobFactory<PrintJob, int, int>::create(10, 10); 
  auto busyJobFactory = JobFactory<BusyJob, int, int>::create(10, 10);

  Task task1(10, 10, 20, 70, std::move(printJobFactory), 1);
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
    scheduler1.displayQueueJobs();
    scheduler1.dispatch();
    sleep(2000);

  }



  return 0;
}

