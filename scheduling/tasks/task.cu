/*
 *  Task class that implements the abstract Task class.
 * */

#include "task.h"
#include <algorithm>

Task::Task(int offset, int compute_time, int rel_deadline, int period,
           std::unique_ptr<JobFactoryBase> jobFactory, int id)
    : offset(offset), compute_time(compute_time), rel_deadline(rel_deadline),
      period(period), jobFactory(std::move(jobFactory)), id(id),
      beginTime(getCurrentTime()) {}

bool Task::isJobReady() {
  float currentTime = getCurrentTime();
  if (!firstJobReleased) {
    return currentTime - beginTime >= offset;
  } else {
    return currentTime - previousJobRelease >= period;
  }
}

Job *Task::releaseJob() {
  if (!firstJobReleased) {
    firstJobReleased = true;
  }
  // get a random absolute deadline.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> realDist(1.0, this->rel_deadline);

  // get a random absolute deadline for the job.
  float absoluteDeadline = realDist(gen);

  // create a new job using the factory.
  this->activeJobs.push_back(this->jobFactory->createJob());

  Job *job = this->activeJobs.back().get();

  job->setParentTask(this);

  float currentTime = getCurrentTime();
  // set the job's absolute deadline.
  job->setAbsoluteDeadline(currentTime + absoluteDeadline);
  // set the most recent job releae time. This is needed because tasks release
  // jobs periodically. The previous job's release time is used to check if
  // enough time has passed for a new job to be ready.
  this->previousJobRelease = currentTime;

  return job;
}

int Task::get_offset() { return this->offset; }

int Task::get_compute_time() { return this->compute_time; }

int Task::get_rel_deadline() { return this->rel_deadline; }

int Task::get_period() { return this->period; }

int Task::get_id() { return this->id; }

float Task::getBeginTime() { return this->beginTime; }

// loop over the list and look for the job to be deleted. Since the list
// contains unique pointers, erasing the pointer from the list will trigger the
// clean up of the job instance. The job is not needed after execution so this
// is safe to do.
void Task::cleanUpJob(Job *job) {
  auto iterator = std::find_if(activeJobs.begin(), activeJobs.end(),
                               [job](auto &uptr) { return uptr.get() == job; });
  if (iterator != this->activeJobs.end()) {
    this->activeJobs.erase(iterator);
  }
}
