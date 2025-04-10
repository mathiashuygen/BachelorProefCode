#include "scheduler.h"

void BaseScheduler::setJobTPCMask(int amountOfTPCs, Job *job) {
  std::vector<MaskElement> masks = DeviceInfo::getDeviceProps()->getTPCMasks();

  int amountOfFreeTPCsFound = 0;

  while (amountOfFreeTPCsFound < amountOfTPCs) {
    MaskElement element = masks.back();
    masks.pop_back();
    if (element.isFree()) {
      // add the mask element to the job's vector of mask elements.
      job->addMask(element);
      // disable the TPC since it is now part of the mask of a job.
      DeviceInfo::getDeviceProps()->disableTPC(element.getIndex());
      amountOfFreeTPCsFound += 1;
    }
  }
}
