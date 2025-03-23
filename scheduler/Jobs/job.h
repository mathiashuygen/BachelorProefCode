/*
 *  Task class that implements the abstract class.
 * */

#include <tuple>
#include <utility>
#include <any>




#ifndef JOB_H
#define JOB_H

class Job{

  private:
    float releaseTime, maximalExecutionTime, absoluteDeadline;


  public:
    //On NVIDIA GPUs, the amount of TPCs allocated to a single kernel can be set. These are the equivalent of the CPU cores allocated
    //to a specific job. 
    int minimumTPCs, maximumTPCs;
    //run time information of a job. Gets defined when a task releases a job. 
    
    //method that has to be overridden by the derrived classes. 
    virtual void launchJob() = 0;

    //this template lets the execute function be completely general. The amount of arguments and the type of the arguments
    //can be chosen by the class that derrives from this one. This makes the class very flexible. 
    template<typename... Args>
    void execute(Args&&... args){
      this->args = std::make_tuple(std::forward<Args>(args)...);
      launchJob();
      return;
    }


    void setReleaseTime(float time){
      this->releaseTime = time;
    }

    void setMaximumExecutionTime(float time){
      this->maximalExecutionTime = time;
    }

    void setAbsoluteDeadline(float time){
      this->absoluteDeadline = time; 
    }

    float getReleaseTime(float time){
      return this->releaseTime;
    }

    float getMaximumExecutionTime(){
      return this->maximalExecutionTime;
    }

    float getAbsoluteDeadline(){
      return this->absoluteDeadline;
    }


  protected:
    std::any args;

};










#endif // JOB_H
