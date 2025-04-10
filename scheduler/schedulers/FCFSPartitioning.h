#include "../Jobs/jobObserver.h"
#include "scheduler.h"
#include <iostream>
#include <queue>

class FCFSPartitioning : public BaseScheduler, public JobObserver {
private:
  std::queue<Job *> jobQueue;

public:
  void onJobCompletion(Job *job, float jobCompletionTime) override;
  void dispatch() override;
  void addJob(Job *job) override;
};
