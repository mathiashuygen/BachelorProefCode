/*
 *  Abstract job class.
 * */

#ifndef JOB_H
#define JOB_H

#include "../../commonLib/libsmctrl/libsmctrl.h"
#include "../common/deviceProps.h"
#include "jobObserver.h"
#include <any>
#include <cstdint>
#include <memory>
#include <sys/types.h>
#include <tuple>
#include <utility>

class JobObserver;

class Job {

private:
  float releaseTime, maximalExecutionTime, absoluteDeadline;
  std::vector<MaskElement> TPCMasks;

protected:
  // On NVIDIA GPUs, the amount of TPCs allocated to a single kernel can be set.
  // These are the equivalent of the CPU cores allocated to a specific job.
  int minimumTPCs, maximumTPCs;
  int threadBlocks, threadsPerBlock;
  static JobObserver *observer;

public:
  // run time information of a job. Gets defined when a task releases a job.

  // method that has to be overridden by the derrived classes.
  virtual void execute() = 0;
  void setReleaseTime(float time);

  void setMaximumExecutionTime(float time);

  void setAbsoluteDeadline(float time);

  float getReleaseTime();

  float getMaximumExecutionTime();

  float getAbsoluteDeadline();

  int getMinimumTPCs();

  int getMaximumTpcs();

  void setThreadsPerBlock(int threads);

  void setThreadBlocks(int threadBlocks);

  int getThreadsPerBlock();

  int getThreadBlocks();

  void setJobObserver(JobObserver *obs);

  // has to be static to be called from the callback function.
  static void notifyJobCompletion(Job *job);

  void addMask(MaskElement element);

  uint64_t combineMasks();

  void releaseMasks();
};

#endif // JOB_H
