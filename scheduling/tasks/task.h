/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC
 * task.
 *
 */

#ifndef TASK_H
#define TASK_H

#include "../common/helpFunctions.h"
#include "../jobs/jobBase/job.h"
#include "../jobs/jobFactory/jobFactory.h"
#include <list>
#include <memory>
#include <random>

class Task {
private:
  float offset, compute_time, rel_deadline, period, beginTime,
      previousJobRelease;
  int id;
  bool firstJobReleased = false;
  std::unique_ptr<JobFactoryBase> jobFactory;
  // job vector. Used to have ownership of the job until it is completed.
  std::vector<std::unique_ptr<Job>> activeJobs;

public:
  Task(int offset, int compute_time, int rel_deadline, int period,
       std::unique_ptr<JobFactoryBase> jobFactory, int id);
  bool isJobReady();
  Job *releaseJob();
  int get_offset();
  int get_compute_time();
  int get_rel_deadline();
  int get_period();
  int get_id();
  float getBeginTime();
  Job *getJob() const;

  void cleanUpJob(Job *job);
};

#endif // TASK_H
