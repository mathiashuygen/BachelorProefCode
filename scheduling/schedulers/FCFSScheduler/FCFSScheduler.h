#ifndef FCFS_SCHEDULER_H
#define FCFS_SCHEDULER_H

#include "../schedulerBase/scheduler.h"
#include <iostream>
#include <queue>

class FCFSScheduler : public BaseScheduler, public JobObserver {
private:
  std::queue<Job *> jobQueue;

public:
  void onJobCompletion(Job *job, float jobCompletionTime) override;
  void dispatch() override;
  void addJob(Job *job) override;
};

#endif
