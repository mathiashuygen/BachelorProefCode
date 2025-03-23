#include "../Jobs/job.h"

#ifndef BASE_SCHEDULER_H
#define BASE_SCHEDULER_H

class BaseScheduler{

  protected:
    int deviceTPCs, SMsPerTPC;
  


  public:
    virtual void dispatch() = 0;
    virtual void addJob(Job* job) = 0;
};

#endif
