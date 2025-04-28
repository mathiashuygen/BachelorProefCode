// JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique
// priority at launch time. Priorities can overlap, it is left to the scheduler
// on how to deal with this.

#include "../../tasks/task.h"
#include "JLFP.h"

JLFP::jobQueue JLFP::createNewJobQueue(Job *job) {
  std::queue<Job *> jobs;
  jobs.push(job);
  jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
  return jobqueue;
}

void JLFP::dispatch() {
  // vector of skipped jobs = jobs that could not be scheduled.
  std::vector<Job *> skippedJobs;
  while (!priorityQueue.empty()) {
    // loop over the queues in a decreasing priority order.
    std::queue<Job *> currJobQueue = priorityQueue.back().jobs;
    while (!currJobQueue.empty()) {
      int TPCsInUse = DeviceInfo::getDeviceProps()->TPCsInUse();
      Job *currJob = currJobQueue.front();
      // case where a job needs more TPCs than there are on the device, give it
      // 1/2 of the TPCs.
      const int N = currJob->getNeededTPCs();
      const int T = DeviceInfo::getDeviceProps()->getTotalTPCsOnDevice();
      if (T < N && N < (2 * T)) {
        int neededTPCs = ceil((float)T / (float)(2 * this->TPC_denom));
        // if there aren't enough TPCs available, have the job wait until enough
        // of them free up.
        if (DeviceInfo::getTotalTPCsOnDevice() - TPCsInUse < neededTPCs) {
          continue;
        }
        // else pop the job and launch it.
        currJobQueue.pop();
        currJob->setJobObserver(this);
        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
        // std::cout << "launched a job that needs more than all the TPCs on the
        // "
        //              "device but less than twice that number.\n";
      }
      // case where the jobs needs more than twice the amount of TPCs on the
      // device, give it all of them to make sure it finishes quickly.
      else if (2 * T <= N) {
        // assign it all the TPCs.
        int neededTPCs = ceil((float)T / (float)this->TPC_denom);
        // if there are TPCs in use, the job has to wait for all the TPCs to
        // free up.
        if (TPCsInUse > 0) {
          continue;
        }
        currJobQueue.pop();
        currJob->setJobObserver(this);
        setJobTPCMask(neededTPCs, currJob);

        currJob->execute();
        // std::cout << "launched a job that needs more than twice the amount of
        // "
        //             "TPCs present on the GPU.\n";
      }
      /* if there aren't enough TPCs for the job at the front of the queue to
       execute, wait for any to free up. the inner queues priority is time
       based, so the first job present in the queue gets to highest priority,
       even though all jobs in this inner queue have the same deadline
       priority.
      */
      else if (N + TPCsInUse <= T) {
        currJobQueue.pop();
        int neededTPCs = ceil((float)N / (float)this->TPC_denom);
        // set the observer. Used to notify the scheduler of the job's
        // completion. When a job is finished with executing its kernel, the job
        // notifies to scheduler which will perform a clean-up.
        currJob->setJobObserver(this);

        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
        // std::cout << "launched a job\n";
      }
      // what if there aren't enough TPCs available that the job requests but
      // you still launch the job on less TPCs => mouldable job.
      else if (TPCsInUse < N && TPCsInUse > 0) {
        int neededTPCs = ceil(TPCsInUse * this->TPC_subset);

        currJob->setJobObserver(this);
        setJobTPCMask(neededTPCs, currJob);
        currJob->execute();
      }
      // if the job is not schedulable, skip it and add it back to the queue at
      // the end of the method.
      else {
        skippedJobs.push_back(currJob);
        currJobQueue.pop();
      }
    }
    // pop the level from the queue.
    priorityQueue.pop_back();
    // add the skipped jobs back to the queue.
    for (Job *job : skippedJobs) {
      std::cout << "added skipped job back" << std::endl;
      this->addJob(job);
    }
  }
}

void JLFP::onJobCompletion(Job *job, float jobCompletionTime) {
  // check if the job met its deadline.
  job->releaseMasks();
  if (jobCompletionTime > job->getAbsoluteDeadline()) {
    this->incDeadlineMisses();
  }
  this->incJobsCompleted();
  // makes sure the job is cleaned up by thet task.
  job->getParentTask()->cleanUpJob(job);
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

JLFP::JLFP(int TPC_denom, float TPC_subset) {
  this->TPC_denom = TPC_denom;
  this->TPC_subset = TPC_subset;
  // init the thread.
  this->running = true;
  this->cleanUpThread = std::thread([this] { this->cleanUpLoop(); });
}

void JLFP::cleanUpLoop() {
  while (this->running) {
    CompletionEvent event;
    if (CompletionQueue::getCompletionQueue().pop(event)) {
      if (!this->running) {
        break;
      }
      delete (event.jobLaunchInfo);
      this->onJobCompletion(event.job, event.completionTime);
      // delete the heap allocated launch info instance.
    } else {
      // pop returned false, means shutdown.
      break;
    }
  }
}

JLFP::~JLFP() {
  this->running = false;
  CompletionQueue::getCompletionQueue().shutdown();
  if (this->cleanUpThread.joinable()) {
    this->cleanUpThread.join();
  }
}
