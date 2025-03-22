#include <iostream>
#include <chrono>
#include <thread>


#ifndef HELP_FUNCTION_H
#define HELP_FUNCTION_H 



inline void sleep(int miliSeconds){
  std::this_thread::sleep_for(std::chrono::milliseconds(miliSeconds));

}


inline double getCurrentTime() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}

#endif 
