/*
 *  Task class that implements the abstract class.
 * */

#include "task.h"

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
  // get a random absolute deadline.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> realDist(1.0, this->rel_deadline);

  float absoluteDeadline = realDist(gen);
  // create a new job using the factory.
  this->job = this->jobFactory->createJob();

  this->job->setAbsoluteDeadline(1.0);
  // reset the begin time;
  this->beginTime = getCurrentTime();

  return this->job.get();
}

int Task::get_offset() { return this->offset; }

int Task::get_compute_time() { return this->compute_time; }

int Task::get_rel_deadline() { return this->rel_deadline; }

int Task::get_period() { return this->period; }

int Task::get_id() { return this->id; }

float Task::getBeginTime() { return this->beginTime; }

Job *Task::getJob() const { return this->job.get(); }
