#include "../../tasks/task.h"
#include "job.h"

void Job::setMaximumExecutionTime(float time) {
  this->maximalExecutionTime = time;
}

void Job::setAbsoluteDeadline(float time) { this->absoluteDeadline = time; }

void Job::setReleaseTime(float time) { this->releaseTime = time; }

float Job::getAbsoluteDeadline() { return this->absoluteDeadline; }

float Job::getMaximumExecutionTime() { return this->maximalExecutionTime; }

float Job::getReleaseTime() { return this->releaseTime; }

int Job::getNeededTPCs() { return this->neededTPCs; }

void Job::setThreadsPerBlock(int threads) { this->threadsPerBlock = threads; }

void Job::setThreadBlocks(int threadBlocks) {
  this->threadBlocks = threadBlocks;
}

int Job::getThreadsPerBlock() { return this->threadsPerBlock; }

int Job::getThreadBlocks() { return this->threadBlocks; }

JobObserver *Job::observer = nullptr;

void Job::setJobObserver(JobObserver *jobObserver) { observer = jobObserver; }

void Job::notifyJobCompletion(Job *job, float jobCompletionTime) {
  if (observer) {
    observer->onJobCompletion(job, jobCompletionTime);
  }
}

void Job::addMask(MaskElement element) { this->TPCMasks.push_back(element); }

uint64_t Job::combineMasks() {
  std::vector<MaskElement> masks = this->TPCMasks;

  // base mask that will be changed to a combination of all the TPC masks.
  uint64_t fullMask = 0xFFFFFFFF;
  while (!masks.empty()) {
    uint64_t singleMask = masks.back().getMask();
    masks.pop_back();
    fullMask = fullMask & singleMask;
  }
  return fullMask;
}

// after the job has completed it should release the TPCs it was using. This is
// done by freeing all the mask elements it used.
void Job::releaseMasks() {
  while (!this->TPCMasks.empty()) {
    MaskElement element = this->TPCMasks.back();
    this->TPCMasks.pop_back();
    DeviceInfo::getDeviceProps()->enableTPC(element.getIndex());
  }
}

void Job::setParentTask(Task *task) { this->parent = task; }

Task *Job::getParentTask() { return this->parent; }

std::string Job::getMessage() { return "standard\n"; }
