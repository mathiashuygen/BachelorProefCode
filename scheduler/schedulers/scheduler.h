
#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H
#include "../Jobs/job.h"
class BaseScheduler {

protected:
  int deviceTPCs, SMsPerTPC, TPCsInUse;

public:
  virtual void dispatch() = 0;
  virtual void addJob(Job *job) = 0;
  virtual ~BaseScheduler() = default;
};

#endif
