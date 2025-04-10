
#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H
#include "../../jobs/jobBase/job.h"
class BaseScheduler {

protected:
  int TPCsInUse;

public:
  virtual void dispatch() = 0;
  virtual void addJob(Job *job) = 0;
  virtual ~BaseScheduler() = default;
  void setJobTPCMask(int amountOfTPCs, Job *job);
};

#endif
