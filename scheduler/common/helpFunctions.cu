#include "helpFunctions.h"



void sleep(int miliSeconds){
  std::this_thread::sleep_for(std::chrono::milliseconds(miliSeconds));

}


double getCurrentTime() {
  using namespace std::chrono;
  return duration_cast<duration<double>>(steady_clock::now().time_since_epoch()).count();
}


