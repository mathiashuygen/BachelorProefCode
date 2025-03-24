//JOB LEVEL FIXED PRIORITY scheduler. This means all jobs have their own unique priority at launch time. 
//Priorities can overlap, it is left to the scheduler on how to deal with this.

#include <cinttypes>
#include <iostream>
#include <memory>
#include <queue>
#include "../Jobs/job.h"
#include <cuda_runtime.h>
#include "scheduler.h"


#ifndef JLFP_H
#define JLFP_H



class JLFP: public BaseScheduler{
  

  private:
    
    //struct that indicated the priority level of a job queue. Easy way to access the priority level of the queue. Otherwise a peak
    //at the first element of the queue was necessary. Added for more structure and readable.
    struct jobQueue{
      //the priority level will be equal to the absolute deadline. The smaller the deadline the higher the priority of the job is. 
      int priorityLevel;
      //actual queue of jobs with the same absolute deadline.
      std::queue<Job*> jobs;

      jobQueue(int level, std::queue<Job*> jobs): priorityLevel(level), jobs(jobs) {}
    };

    /*vector that contains structs that contain queues of jobs. Every queue has its own priority level. 
    Earliest Deadline First priority assignment. 
    */
    std::vector<jobQueue> priorityQueue;
    
    
    jobQueue createNewJobQueu(Job* job){
      std::queue<Job*> jobs;
      jobs.push(job);
      jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
      return jobqueue;
    }


  public:
    void dispatch() override{
      while(!priorityQueue.empty()){
        //TODO
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
    void addJob(Job* job) override{
      if(priorityQueue.empty()){
        jobQueue jobqueue = createNewJobQueu(job);
        priorityQueue.push_back(jobqueue);
        //std::cout<<"queue was empty, thus added at to the front with level: "<<job->getAbsoluteDeadline()<<"\n";
        return;
      }

      for(size_t i = 0; i < priorityQueue.size(); ++i){
        jobQueue currJobqueue = priorityQueue.at(i);
        float jobAbsoluteDeadline = job->getAbsoluteDeadline();

        if(jobAbsoluteDeadline == currJobqueue.priorityLevel){
          currJobqueue.jobs.push(job);
          //std::cout<<"found matching queue at level: "<<jobAbsoluteDeadline<<" to the job queue\n";
          return;
        }
        else if(i == 0 && jobAbsoluteDeadline > priorityQueue.begin()->priorityLevel){
          jobQueue jobqueue = createNewJobQueu(job); 
          priorityQueue.insert(priorityQueue.begin(), jobqueue);
          //std::cout<<"job with new lowest priority added at level: "<<jobAbsoluteDeadline<<"\n";
          return;

        }
        else if(i == priorityQueue.size() - 1){
          jobQueue jobqueue = createNewJobQueu(job);
          priorityQueue.push_back(jobqueue);
          //std::cout<<"found new highest priority, added to the end with level: "<< jobAbsoluteDeadline <<"\n";
          return;
        }
        else if(i != 0 && jobAbsoluteDeadline < priorityQueue.at(i - 1).priorityLevel && jobAbsoluteDeadline > priorityQueue.at(i).priorityLevel){
          jobQueue jobqueue = createNewJobQueu(job);
          priorityQueue.insert(priorityQueue.begin() + i, jobqueue);
          //std::cout<<"found new priority in the middle of the queues at level: "<<jobAbsoluteDeadline<<"\n";
          return;
        }
      }
    }


    void displayQueuePriorities(){
      for(jobQueue jobqueue : priorityQueue){
        std::cout<<jobqueue.priorityLevel<<", ";
      }
      std::cout<<"\n";
    }



    JLFP(){
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      this->SMsPerTPC = 2;
      this->deviceTPCs = deviceProp.multiProcessorCount/SMsPerTPC;
      this->TPCsInUse = 0;

    }


};




#endif
