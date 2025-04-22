/*
 *
 * Queue that is used by the clean up thread of the schedulers. The scheduler's
 * thread will keep on popping from this queue until no jobs are left to clean
 * up.
 *
 *
 * */

#ifndef COMPLETION_QUEUE_H
#define COMPLETION_QUEUE_H

#include "../../jobs/jobBase/job.h"
#include "../../jobs/jobLaunchInformation/baseLaunchInformation.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

struct CompletionEvent {
  Job *job;
  float completionTime;
  KernelLaunchInfoBase *jobLaunchInfo;
};

class CompletionQueue {
private:
  std::mutex mtx;
  std::condition_variable cv;
  std::queue<CompletionEvent> jobCleanUpQueue;
  // shutdown is requested when scheduler instance is deleted.
  std::atomic<bool> shutdownRequest{false};

public:
  void push(CompletionEvent ev);

  bool pop(CompletionEvent &ev);

  static CompletionQueue &getCompletionQueue();

  void shutdown();
};

#endif
