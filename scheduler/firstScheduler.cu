#include <iostream>
#include <cuda_runtime.h>
#include <thread>
#include "task.cu"
#include "Jobs/printJob.cu"
#include <chrono>









void sleep(int miliSeconds){
  std::this_thread::sleep_for(std::chrono::milliseconds(miliSeconds));

}




int main(){
  
  
  Task t1(1, 2, 3, 4, executeJob, 1);
  
  while(true){
    
    t1.launchJob();
    sleep(1000);
  
  }



  return 0;
}

