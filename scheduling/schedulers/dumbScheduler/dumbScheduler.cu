#include "dumbScheduler.h"

void DumbScheduler::dispatch() {
  while (!this->jobQueue.empty()) {
    Job *currJob = this->jobQueue.front();
    this->jobQueue.pop();
    currJob->setJobObserver(this);
    currJob->execute();
    // std::cout << "launched a job from the dumb scheduler\n";
  }
}

void DumbScheduler::addJob(Job *job) { this->jobQueue.push(job); }

void DumbScheduler::onJobCompletion(Job *job, float jobCompletionTime) {
  // check if the job met its deadline.
  job->releaseMasks();
  const float deadline = job->getAbsoluteDeadline();
  if (deadline < jobCompletionTime) {
    this->incDeadlineMisses();
  }
  this->incJobsCompleted();
}

DumbScheduler::DumbScheduler() {
  // init the thread.
  this->running = true;
  this->cleanUpThread = std::thread([this] { this->cleanUpLoop(); });
}

void DumbScheduler::cleanUpLoop() {
  while (this->running) {
    CompletionEvent event;
    if (CompletionQueue::getCompletionQueue().pop(event)) {
      if (!this->running) {
        break;
      }
      delete (event.jobLaunchInfo);
      this->onJobCompletion(event.job, event.completionTime);
      // delete the heap allocated launch info instance.
    } else {
      // pop returned false, means shutdown.
      break;
    }
  }
}

DumbScheduler::~DumbScheduler() {
  this->running = false;
  CompletionQueue::getCompletionQueue().shutdown();
  if (this->cleanUpThread.joinable()) {
    this->cleanUpThread.join();
  }
}
