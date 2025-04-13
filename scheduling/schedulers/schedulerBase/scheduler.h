#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H
#include "../../jobs/jobBase/job.h"
class BaseScheduler {

protected:
  int TPCsInUse = 0;
  int jobsCompleted = 0;
  int deadlineMisses = 0;

public:
  virtual void dispatch() = 0;
  virtual void addJob(Job *job) = 0;
  virtual ~BaseScheduler() = default;
  void setJobTPCMask(int amountOfTPCs, Job *job);

  int getJobsCompleted();
  int getDeadlineMisses();
  void incJobsCompleted();
  void incDeadlineMisses();
};

#endif
