//JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique priority at launch time. 
//Priorities can overlap, it is left to the scheduler on how to deal with this.



#include "JLFP.h"


JLFP::jobQueue JLFP::createNewJobQueu(Job* job){
  std::queue<Job*> jobs;
  jobs.push(job);
  jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
  return jobqueue;
}


void JLFP::dispatch(){
  while(!priorityQueue.empty()){
    //loop over the queues in a decreasing priority order.
    std::queue<Job*> currJobQueue = priorityQueue.back().jobs; 
    while(!currJobQueue.empty()){
      //if there are no remaining TPCs, wait for any to free up, also check if there are enough available TPCs left
      //to execute the job's kernel.
      if(this->TPCsInUse < this->deviceTPCs && currJobQueue.front()->maximumTPCs + this->TPCsInUse <= this->deviceTPCs){
        Job* currJob = currJobQueue.front();
        currJobQueue.pop();
        this->TPCsInUse = this->TPCsInUse + currJob->maximumTPCs;
        currJob->execute(10, 10, 10000);
        std::cout<<"launched a job's kernel\n";
        this->TPCsInUse = this->TPCsInUse - currJob->maximumTPCs;
      }
    }
    priorityQueue.pop_back();
  }
}


/*
* Adding a job to the vector has to take care of multiple scenarios:
*
*  -If the vector is empty, a new queue has to be inserted
*   into the vector.
*
*  -If the job that has to be added has a higher priority than any job queues so far, a new job queue has to be 
*   created and inserted at the end of the vector.
*
*  -If the job that has to be added has a lower priority that any job in the queues so far, a new job queue has to be 
*   created and inserted to the beginning of the vector.
*
*  -If the job has a priority that has never been seen before which isn't the lowest of highest one, a new job queue 
*   has to created and inserted in the middle of the vector between to priorities that are respectively lower and higher 
*   than the new job's one. 
*
* */
void JLFP::addJob(Job* job){
  if(priorityQueue.empty()){
    jobQueue jobqueue = createNewJobQueu(job);
    priorityQueue.push_back(jobqueue);
    std::cout<<"queue was empty, thus added at to the front with level: "<<job->getAbsoluteDeadline()<<"\n";
    return;
  }

  for(size_t i = 0; i < priorityQueue.size(); ++i){
    jobQueue currJobqueue = priorityQueue.at(i);
    float jobAbsoluteDeadline = job->getAbsoluteDeadline();

    if(jobAbsoluteDeadline == currJobqueue.priorityLevel){
      priorityQueue.at(i).jobs.push(job);
      std::cout<<"found matching queue at level: "<<jobAbsoluteDeadline<<" to the job queue\n";
      return;
    }
    else if(i == 0 && jobAbsoluteDeadline > priorityQueue.begin()->priorityLevel){
      jobQueue jobqueue = createNewJobQueu(job); 
      priorityQueue.insert(priorityQueue.begin(), jobqueue);
      std::cout<<"job with new lowest priority added at level: "<<jobAbsoluteDeadline<<"\n";
      return;
    }
    else if(i == priorityQueue.size() - 1){
      jobQueue jobqueue = createNewJobQueu(job);
      priorityQueue.push_back(jobqueue);
      std::cout<<"found new highest priority, added to the end with level: "<< jobAbsoluteDeadline <<"\n";
      return;
    }
    else if(i != 0 && jobAbsoluteDeadline < priorityQueue.at(i - 1).priorityLevel && jobAbsoluteDeadline > priorityQueue.at(i).priorityLevel){
      jobQueue jobqueue = createNewJobQueu(job);
      priorityQueue.insert(priorityQueue.begin() + i, jobqueue);
      std::cout<<"found new priority in the middle of the queues at level: "<<jobAbsoluteDeadline<<"\n";
      return;
    }
  }
}


void JLFP::displayQueuePriorities(){
  for(jobQueue jobqueue : priorityQueue){
    std::cout<<jobqueue.priorityLevel<<", ";
  }
  std::cout<<"\n";
}

void JLFP::displayQueueJobs(){
  for(jobQueue jobqueue : priorityQueue){
    std::queue<Job*> currJobs = jobqueue.jobs;
    while(!currJobs.empty()){
      std::cout<<"job with priority: "<<currJobs.front()->getAbsoluteDeadline()<<"\n";
      currJobs.pop();
    }
  }
}



JLFP::JLFP(){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  this->SMsPerTPC = 2;
  this->deviceTPCs = deviceProp.multiProcessorCount/SMsPerTPC;
  this->TPCsInUse = 0;
}





