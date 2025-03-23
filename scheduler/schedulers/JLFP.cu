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
      //the priority level will be equal to the deadline. The smaller the deadline the higher the priority of the job is. 
      int priorityLevel;
      std::queue<Job*> jobs;

      jobQueue(int level, std::queue<Job*> jobs): priorityLevel(level), jobs(jobs) {}
    };

    /*queue that holds all of the jobs that are ready to be disptached. The queue contains other queues that reflect the priority level.
    the queue at the head of the job queue contains all the jobs that are ready to be dispatched that are of the highest priority.
    Earliest Deadline First priority assignment. 
    */
    std::vector<jobQueue> priorityQueue;
    
    //More modern NVIDIA GPUs contain TPCs that each contain 2 SMs.
    
  public:
    void dispatch() override{
      while(!priorityQueue.empty()){

      }
    }

    void addJob(Job* job) override{
      if(priorityQueue.empty()){
        std::queue<Job*> jobs; 
        jobs.push(job);
        jobQueue jobqueue(job->getAbsoluteDeadline(), jobs);
        priorityQueue.push_back(jobqueue);
        std::cout<<"queue was empty, thus added at to the front with level: "<<job->getAbsoluteDeadline()<<"\n";
        return;
      }

      for(size_t i = 0; i < priorityQueue.size(); ++i){
        jobQueue jobqueue = priorityQueue.at(i);
        float jobAbsoluteDeadline = job->getAbsoluteDeadline();

        if(jobAbsoluteDeadline == jobqueue.priorityLevel){
          jobqueue.jobs.push(job);
          std::cout<<"found matching queue at level: "<<jobAbsoluteDeadline<<" to the job queue\n";
          return;
        }
        else if(i == 0 && jobAbsoluteDeadline > priorityQueue.begin()->priorityLevel){
          std::queue<Job*> jobs; 
          jobs.push(job);
          jobQueue jobqueue(jobAbsoluteDeadline, jobs);
          priorityQueue.insert(priorityQueue.begin(), jobqueue);
          std::cout<<"job with new lowest priority added at level: "<<jobAbsoluteDeadline<<"\n";
          return;

        }
        else if(i == priorityQueue.size() - 1){
          std::queue<Job*> jobs; 
          jobs.push(job);
          jobQueue jobqueue(jobAbsoluteDeadline, jobs);
          priorityQueue.push_back(jobqueue);
          std::cout<<"found new highest priority, added to the end with level: "<< jobAbsoluteDeadline <<"\n";
          return;
        }
        else if(i != 0 && jobAbsoluteDeadline < priorityQueue.at(i - 1).priorityLevel && jobAbsoluteDeadline > priorityQueue.at(i).priorityLevel){
          std::queue<Job*> jobs; 
          jobs.push(job);
          jobQueue jobqueue(jobAbsoluteDeadline, jobs);
          priorityQueue.insert(priorityQueue.begin() + i, jobqueue);
          std::cout<<"found new priority in the middle of the queues at level: "<<jobAbsoluteDeadline<<"\n";
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

    }


};




#endif
