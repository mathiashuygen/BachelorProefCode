// JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique
// priority at launch time. Priorities can overlap, it is left to the scheduler
// on how to deal with this.

#include "JLFP.h"

JLFP::jobQueue JLFP::createNewJobQueue(Job *job) {
  std::queue<Job *> jobs;
  jobs.push(job);
  jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
  return jobqueue;
}

void JLFP::dispatch() {
  while (!priorityQueue.empty()) {
    // loop over the queues in a decreasing priority order.
    std::queue<Job *> currJobQueue = priorityQueue.back().jobs;
    while (!currJobQueue.empty()) {
      Job *currJob = currJobQueue.front();
      // case where a job needs more TPCs than there are on the device, give it
      // 1/2 of the TPCs.
      if (currJob->getNeededTPCs() >
              DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice() &&
          currJob->getNeededTPCs() <
              2 * DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice()) {
        int neededTPCs =
            DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice() / 2;
        // if there aren't enough TPCs available, have the job wait until enough
        // of them free up.
        if (DeviceInfo::getTotalTPCsOnDevice() - TPCsInUse < neededTPCs) {
          continue;
        }
        // else pop the job and launch it.
        currJobQueue.pop();
        this->TPCsInUse = neededTPCs;
        currJob->setJobObserver(this);
        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
        std::cout << "launched a job that needs more than all the TPCs on the "
                     "device but less than twice that number.\n";
      }
      // case where the jobs needs more than twice the amount of TPCs on the
      // device, give it all of them to make sure it finishes quickly.
      else if (currJob->getNeededTPCs() >=
               2 * DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice()) {
        // assign it all the TPCs.
        int neededTPCs = DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice();
        // if there are TPCs in use, the job has to wait for all the TPCs to
        // free up.
        if (this->TPCsInUse > 0) {
          continue;
        }
        currJobQueue.pop();
        this->TPCsInUse = neededTPCs;
        currJob->setJobObserver(this);
        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
        std::cout << "launched a job that needs more than twice the amount of "
                     "TPCs present on the GPU.\n";
      }
      /* if there aren't enough TPCs for the job at the front of the queue to
       execute, wait for any to free up. the inner queues priority is time
       based, so the first job present in the queue gets to highest priority,
       even though all jobs in this inner queue have the same deadline
       priority.
      */
      else if (currJob->getNeededTPCs() + this->TPCsInUse <=
               DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice()) {
        currJobQueue.pop();
        int neededTPCs = currJob->getNeededTPCs();
        this->TPCsInUse = this->TPCsInUse + neededTPCs;
        // set the observer. Used to notify the scheduler of the job's
        // completion. When a job is finished with executing its kernel, the job
        // notifies to scheduler which will perform a clean-up.
        currJob->setJobObserver(this);

        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
        std::cout << "launched a job\n";
      }
    }
    // pop the level from the queue.
    priorityQueue.pop_back();
  }
}

void JLFP::onJobCompletion(Job *job, float jobCompletionTime) {
  this->TPCsInUse -= job->getNeededTPCs();
  job->releaseMasks();
  if (job->getReleaseTime() + job->getAbsoluteDeadline() < jobCompletionTime) {
    std::cout << "job finished on time\n";
  } else {
    std::cout << "job missed its deadline\n";
  }
  // std::cout << "job finished execution\n";
}

/*
 * Adding a job to the vector has to take care of multiple scenarios:
 *
 *  -If the vector is empty, a new queue has to be inserted
 *   into the vector.
 *
 *  -If the job that has to be added has a higher priority than any of the job
 * queues so far, a new job queue has to be created and inserted at the end of
 * the vector.
 *
 *  -If the job that has to be added has a lower priority that any job in the
 * queues so far, a new job queue has to be created and inserted to the
 * beginning of the vector.
 *
 *  -If the job has a priority that has never been seen before which isn't the
 * lowest or highest one, a new job queue has to created and inserted in the
 * middle of the vector between to priorities that are lower and
 * higher than the new job's one.
 *
 * */
void JLFP::addJob(Job *job) {
  if (priorityQueue.empty()) {
    jobQueue jobqueue = createNewJobQueue(job);
    priorityQueue.push_back(jobqueue);
    // std::cout<<"queue was empty, thus added to the front with level:
    // "<<job->getAbsoluteDeadline()<<"\n";
    return;
  }

  for (size_t i = 0; i < priorityQueue.size(); ++i) {
    jobQueue currJobqueue = priorityQueue.at(i);
    float jobAbsoluteDeadline = job->getAbsoluteDeadline();

    if (jobAbsoluteDeadline == currJobqueue.priorityLevel) {
      priorityQueue.at(i).jobs.push(job);
      // std::cout<<"found matching queue at level: "<<jobAbsoluteDeadline<<" to
      // the job queue\n";
      return;
    } else if (i == 0 &&
               jobAbsoluteDeadline > priorityQueue.begin()->priorityLevel) {
      jobQueue jobqueue = createNewJobQueue(job);
      priorityQueue.insert(priorityQueue.begin(), jobqueue);
      // std::cout<<"job with new lowest priority added at level:
      // "<<jobAbsoluteDeadline<<"\n";
      return;
    } else if (i == priorityQueue.size() - 1) {
      jobQueue jobqueue = createNewJobQueue(job);
      priorityQueue.push_back(jobqueue);
      // std::cout<<"found new highest priority, added to the end with level:
      // "<< jobAbsoluteDeadline <<"\n";
      return;
    } else if (i != 0 &&
               jobAbsoluteDeadline < priorityQueue.at(i - 1).priorityLevel &&
               jobAbsoluteDeadline > priorityQueue.at(i).priorityLevel) {
      jobQueue jobqueue = createNewJobQueue(job);
      priorityQueue.insert(priorityQueue.begin() + i, jobqueue);
      // std::cout<<"found new priority in the middle of the queues at level:
      // "<<jobAbsoluteDeadline<<"\n";
      return;
    }
  }
}

void JLFP::displayQueuePriorities() {
  for (jobQueue jobqueue : priorityQueue) {
    std::cout << jobqueue.priorityLevel << ", ";
  }
  std::cout << "\n";
}

void JLFP::displayQueueJobs() {
  for (jobQueue jobqueue : priorityQueue) {
    std::queue<Job *> currJobs = jobqueue.jobs;
    while (!currJobs.empty()) {
      std::cout << "job with priority: "
                << currJobs.front()->getAbsoluteDeadline() << "\n";
      currJobs.pop();
    }
  }
}

JLFP::JLFP() { this->TPCsInUse = 0; }
