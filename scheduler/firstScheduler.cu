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
  
  Task<int, int> task(10, 5, 20, 100, printJobExec, 1, std::chrono::system_clock::now(), std::chrono::system_clock::now());
  
  while(true){
    
    task.execute(1, 1);

    sleep(1000);
  }



  return 0;
}

