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
  
  Task<int, int, int> task1(10, 5, 20, 100, printJobExec, 1, std::chrono::system_clock::now(), std::chrono::system_clock::now());
  Task<int, int, int> task2(10, 5, 20, 100, printJobExec, 2, std::chrono::system_clock::now(), std::chrono::system_clock::now());

  
  while(true){
    
    task1.execute(1, 1, 1000000);
    task2.execute(2, 1, 10);


    sleep(3000);
  }



  return 0;
}

