#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "../common/helpFunctions.h"
#include "../Jobs/job.h"
/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC task.
 *
 */



/*
 *  Task class that implements the abstract class.
 * */
class Task{
  private:
    float offset, compute_time, rel_deadline, period, beginTime, previousJobRelease; 
    int id;
    //time info used to adhere to the periodicity of a task.
    //function that does not return anything and takes any amount of args with any type. 
    std::unique_ptr<Job> job;
    //The first job releases at a specifiec time. Used in the check if the first job can be released.
    bool firstJobReleased = false;
    

  public:

    Task(int offset, int compute_time, int rel_deadline, int period, std::unique_ptr<Job> job, int id)
      :offset(offset), compute_time(compute_time), rel_deadline(rel_deadline), period(period), job(std::move(job)), id(id), beginTime(getCurrentTime()) {
      

      }
    

    bool isJobReady(){
      float currentTime = getCurrentTime();
      if(!firstJobReleased){

        std::cout<<"currentTime: "<<currentTime<<"\n"; 
        std::cout<<"beginTime: "<<beginTime<<"\n";
        std::cout<<"offset: "<<this->offset<<"\n\n";
        return currentTime - beginTime >= offset;
      
      }
      else{
        return currentTime - previousJobRelease >= period;
      }
    }

    int get_offset(){
      return this->offset;
    }

    int get_compute_time(){
      return this->compute_time;
    }

    int get_rel_deadline(){
      return this->rel_deadline;
    }

    int get_period(){
      return this->period;
    }

    int get_id(){
      return this->id;
    }

    float getBeginTime(){
      return this->beginTime;
    }


    Job* getJob() const{
      return this->job.get();
    }

   
};
