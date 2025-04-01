// JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique
// priority at launch time. Priorities can overlap, it is left to the scheduler
// on how to deal with this.

#include "JLFP.h"

JLFP::jobQueue JLFP::createNewJobQueu(Job *job) {
  std::queue<Job *> jobs;
  jobs.push(job);
  jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
  return jobqueue;
}

void JLFP::setJobTPCMask(int amountOfTPCs, Job *job) {
  std::vector<MaskElement> masks = DeviceInfo::getDeviceProps()->getTPCMasks();

  int amountOfFreeTPCsFound = 0;

  while (amountOfFreeTPCsFound < amountOfTPCs) {
    MaskElement element = masks.back();
    masks.pop_back();
    if (element.isFree()) {
      // add the mask element to the job's vector of mask elements.
      job->addMask(element);
      // disable the TPC since it is now part of the mask of a job.
      DeviceInfo::getDeviceProps()->disableTPC(element.getIndex());
      amountOfFreeTPCsFound += 1;
    }
  }
}

void JLFP::dispatch() {
  while (!priorityQueue.empty()) {
    // loop over the queues in a decreasing priority order.
    std::queue<Job *> currJobQueue = priorityQueue.back().jobs;
    while (!currJobQueue.empty()) {
      /* if there aren't enough TPCs for the job at the front of the queue to
       execute, wait for any to free up. the inner queues priority is time
       based, so the first job present in the queue gets to highest priority,
       even though all jobs in this inner queue have the same deadline
       priority.
      */
      if (currJobQueue.front()->getNeededTPCs() + this->TPCsInUse <=
          DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice()) {
        Job *currJob = currJobQueue.front();
        currJobQueue.pop();
        this->TPCsInUse = this->TPCsInUse + currJob->getNeededTPCs();
        // set the observer. Used to notify the scheduler of the job's
        // completion. When a job is finished with executing its kernel, the job
        // notifies to scheduler which will perform a clean-up.
        currJob->setJobObserver(this);

        setJobTPCMask(currJob->getNeededTPCs(), currJob);
        currJob->execute();
        std::cout << "launched a job\n";
      }
    }
    priorityQueue.pop_back();
  }
}

void JLFP::onJobCompletion(Job *job) {
  this->TPCsInUse -= job->getNeededTPCs();
  job->releaseMasks();
  std::cout << "job finished execution\n";
}

/*
 * Adding a job to the vector has to take care of multiple scenarios:
 *
 *  -If the vector is empty, a new queue has to be inserted
 *   into the vector.
 *
 *  -If the job that has to be added has a higher priority than any job queues
 * so far, a new job queue has to be created and inserted at the end of the
 * vector.
 *
 *  -If the job that has to be added has a lower priority that any job in the
 * queues so far, a new job queue has to be created and inserted to the
 * beginning of the vector.
 *
 *  -If the job has a priority that has never been seen before which isn't the
 * lowest of highest one, a new job queue has to created and inserted in the
 * middle of the vector between to priorities that are respectively lower and
 * higher than the new job's one.
 *
 * */
void JLFP::addJob(Job *job) {
  if (priorityQueue.empty()) {
    jobQueue jobqueue = createNewJobQueu(job);
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
      jobQueue jobqueue = createNewJobQueu(job);
      priorityQueue.insert(priorityQueue.begin(), jobqueue);
      // std::cout<<"job with new lowest priority added at level:
      // "<<jobAbsoluteDeadline<<"\n";
      return;
    } else if (i == priorityQueue.size() - 1) {
      jobQueue jobqueue = createNewJobQueu(job);
      priorityQueue.push_back(jobqueue);
      // std::cout<<"found new highest priority, added to the end with level:
      // "<< jobAbsoluteDeadline <<"\n";
      return;
    } else if (i != 0 &&
               jobAbsoluteDeadline < priorityQueue.at(i - 1).priorityLevel &&
               jobAbsoluteDeadline > priorityQueue.at(i).priorityLevel) {
      jobQueue jobqueue = createNewJobQueu(job);
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
