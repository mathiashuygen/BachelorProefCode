#include "dumbScheduler.h"

void DumbScheduler::dispatch() {
  while (!this->jobQueue.empty()) {
    Job *currJob = this->jobQueue.front();
    this->jobQueue.pop();
    currJob->setJobObserver(this);
    currJob->execute();
    std::cout << "launched a job from the dumb scheduler\n";
  }
}

void DumbScheduler::addJob(Job *job) { this->jobQueue.push(job); }

void DumbScheduler::onJobCompletion(Job *job, float jobCompletionTime) {
  return;
}
