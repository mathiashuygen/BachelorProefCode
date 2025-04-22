#ifndef FCFS_SCHEDULER_H
#define FCFS_SCHEDULER_H

#include "../asyncCompletionQueue/completionQueue.h"
#include "../schedulerBase/scheduler.h"
#include <iostream>
#include <queue>

class FCFSScheduler : public BaseScheduler, public JobObserver {
private:
  std::queue<Job *> jobQueue;

  // clean up fields and methods.
  // the actual thread.
  std::thread cleanUpThread;
  // boolean that is checked to keep the loop inside the thread running.
  std::atomic<bool> running{false};
  // loop method that loops inside the thread to clean up all the jobs.
  void cleanUpLoop();

public:
  void onJobCompletion(Job *job, float jobCompletionTime) override;
  void dispatch() override;
  void addJob(Job *job) override;
  FCFSScheduler();
  ~FCFSScheduler();
};

#endif
