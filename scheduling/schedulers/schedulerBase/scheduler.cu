#include "scheduler.h"
void BaseScheduler::setJobTPCMask(int amountOfTPCs, Job *job) {
  std::vector<MaskElement> masks = DeviceInfo::getDeviceProps()->getTPCMasks();

  int amountOfFreeTPCsFound = 0;

  while (amountOfFreeTPCsFound < amountOfTPCs) {
    MaskElement element = masks.back();
    if (element.isFree()) {
      // add the mask element to the job's vector of mask elements.
      job->addMask(element);
      masks.pop_back();
      // disable the TPC since it is now part of the mask of a job.
      DeviceInfo::getDeviceProps()->disableTPC(element.getIndex());
      amountOfFreeTPCsFound += 1;
    }
  }
}

int BaseScheduler::getJobsCompleted() { return this->jobsCompleted; }

int BaseScheduler::getDeadlineMisses() { return this->deadlineMisses; }

void BaseScheduler::incJobsCompleted() { this->jobsCompleted += 1; }

void BaseScheduler::incDeadlineMisses() { this->deadlineMisses += 1; }
