/*
 * Observer that allows for a scheduler to observe a job. When a job is done
 * executing, it notifies the observer. Then the observer will perform a
 * clean-up using the information of the observed job. For example, the amount
 * of TPCs or threads in use could be communicated to the scheduler. The
 * scheduler can release the TPCs to other jobs when a completed job notifies
 * the scheduler of its end of execution.
 * */

#ifndef JOB_OBSERVER_H
#define JOB_OBSERVER_H

// forward declaration to have access to the job type.
class Job;

class JobObserver {
public:
  virtual void onJobCompletion(Job *job, float jobCompletionTime) = 0;
  virtual ~JobObserver() = default;
};

#endif
