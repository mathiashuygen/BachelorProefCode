#ifndef BASE_LAUNCH_INFORMATION_H
#define BASE_LAUNCH_INFORMATION_H

class Job;

class KernelLaunchInfoBase {
public:
  Job *jobPtr;

  KernelLaunchInfoBase(Job *job) : jobPtr(job) {}

  virtual ~KernelLaunchInfoBase() = default;
};

#endif
