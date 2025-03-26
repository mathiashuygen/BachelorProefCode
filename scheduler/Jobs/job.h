/*
 *  Task class that implements the abstract class.
 * */

#ifndef JOB_H
#define JOB_H

#include <tuple>
#include <utility>
#include <any>
#include <memory>



class Job{

  private:
    float releaseTime, maximalExecutionTime, absoluteDeadline;

    
  protected:
    int minimumTPCs, maximumTPCs;

  public:
    //On NVIDIA GPUs, the amount of TPCs allocated to a single kernel can be set. These are the equivalent of the CPU cores allocated
    //to a specific job. 
  
    //run time information of a job. Gets defined when a task releases a job. 
    
    //method that has to be overridden by the derrived classes. 
    virtual void execute() = 0;



    void setReleaseTime(float time);

    void setMaximumExecutionTime(float time);

    void setAbsoluteDeadline(float time);

    float getReleaseTime();

    float getMaximumExecutionTime();

    float getAbsoluteDeadline();
    
    int getMinimumTPCs();

    int getMaximumTpcs();

};










#endif // JOB_H
