
#include "../executive/scheduling.h"
#include "../schedulers/FCFSScheduler/FCFSScheduler.h"
#include "../schedulers/JLFPScheduler/JLFP.h"
#include "../schedulers/dumbScheduler/dumbScheduler.h"

// Function to create a scheduler based on type
std::unique_ptr<BaseScheduler>
createScheduler(const std::string &type, int tpcSplitDenom, float tpcSubset) {
  if (type == "JLFP") {
    return std::make_unique<JLFP>(tpcSplitDenom, TPCSubset);
  } else if (type == "FCFS") {
    return std::make_unique<FCFSScheduler>();
  } else if (type == "dumbScheduler") {
    return std::make_unique<DumbScheduler>();
  } else {
    std::cerr << "Unknown scheduler type: " << type << std::endl;
    return nullptr;
  }
}
