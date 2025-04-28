#ifndef JLFP_H
#define JLFP_H
// JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique
// priority at launch time. Priorities can overlap, it is left to the scheduler
// on how to deal with this.
#include "../asyncCompletionQueue/completionQueue.h"
#include "../schedulerBase/scheduler.h"

class JLFP : public BaseScheduler, public JobObserver {
private:
  struct jobQueue {
    int priorityLevel;
    std::queue<Job *> jobs;
    jobQueue(int level, std::queue<Job *> jobs)
        : priorityLevel(level), jobs(jobs) {}
  };

  std::vector<jobQueue> priorityQueue;

  jobQueue createNewJobQueue(Job *job);

  // clean up fields and methods.
  // the actual thread.
  std::thread cleanUpThread;
  // boolean that is checked to keep the loop inside the thread running.
  std::atomic<bool> running{false};
  // loop method that loops inside the thread to clean up all the jobs.
  void cleanUpLoop();

  // divides the amount of TPCS a job needs when assigning a mask. Used for
  // benchmarks.
  int TPC_denom;
  float TPC_subset;

public:
  void onJobCompletion(Job *job, float jobCompletionTime) override;
  void dispatch() override;
  void addJob(Job *job) override;
  void displayQueuePriorities();
  void displayQueueJobs();

  // added constructor to overwrite running boolean.
  JLFP(int TPC_denom, float TPC_subset);
  ~JLFP() override;
};

#endif
