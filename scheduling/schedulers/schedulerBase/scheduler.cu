#include "scheduler.h"

void BaseScheduler::setJobTPCMask(int amountOfTPCs, Job *job) {
  std::vector<MaskElement> masks = DeviceInfo::getDeviceProps()->getTPCMasks();
  int amountOfFreeTPCsFound = 0;
  while (amountOfFreeTPCsFound < amountOfTPCs) {
    if (masks.empty()) {
      // This should not happen if amountOfTPCs is calculated correctly
      // and there are enough free TPCs, but it's a crucial safety check.
      std::cerr
          << "ERROR: Ran out of masks in setJobTPCMask while still needing "
          << (amountOfTPCs - amountOfFreeTPCsFound) << " TPCs!" << std::endl;
      // Handle error appropriately - maybe break, maybe throw, maybe assert
      break;
    }
    MaskElement element = masks.back();
    masks.pop_back();
    int elementIndex = element.getIndex();
    // ... rest of the loop ...
    if (element.isFree()) {
      job->addMask(element);
      DeviceInfo::getDeviceProps()->disableTPC(elementIndex);
      amountOfFreeTPCsFound += 1;
    }
    // Consider adding a check here too: if (!element.isFree()), what happens?
    // Does the loop potentially run forever if there aren't enough *free*
    // masks?
  }
}

int BaseScheduler::getJobsCompleted() { return this->jobsCompleted; }

int BaseScheduler::getDeadlineMisses() { return this->deadlineMisses; }

void BaseScheduler::incJobsCompleted() { this->jobsCompleted += 1; }

void BaseScheduler::incDeadlineMisses() { this->deadlineMisses += 1; }
