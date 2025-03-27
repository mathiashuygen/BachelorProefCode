/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC task.
 *
 */

#ifndef TASK_H
#define TASK_H

#include <memory>
#include <random>
#include "../common/helpFunctions.h"
#include "../Jobs/job.h"
#include "../Jobs/jobFactory.h"

class Task {
private:
    float offset, compute_time, rel_deadline, period, beginTime, previousJobRelease;
    int id;
    bool firstJobReleased = false;
    std::unique_ptr<Job> job;
    std::unique_ptr<JobFactoryBase> jobFactory;

public:
    Task(int offset, int compute_time, int rel_deadline, int period,
         std::unique_ptr<JobFactoryBase> jobFactory, int id);
    bool isJobReady();
    Job* releaseJob();
    int get_offset();
    int get_compute_time();
    int get_rel_deadline();
    int get_period();
    int get_id();
    float getBeginTime();
    Job* getJob() const;
};

#endif // TASK_H
