#include <iostream>
#include <cuda_runtime.h>
#include <functional>
#include <iterator>
#include <tuple>
#include <utility>
#include <any>
#include <chrono>
/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC task.
 *
 */


/*
 * base class to extend. Used this to be able to provide jobs to the task and have them take 
 * any amount of arguments of any type.
 * */
class TaskBase{
  public:
    virtual void launchJob() = 0;


    template<typename... Args>
    void execute(Args&&... args){
      //forward the arguments to the job
      this->args = std::make_tuple(std::forward<Args>(args)...);
      launchJob();
      return;
    }


    virtual ~TaskBase() = default;

  protected:
    std::any args;


};

/*
 *  Task class that implements the abstract class.
 * */
template<typename... FuncArgs>
class Task: public TaskBase{
  private:
    int offset, compute_time, rel_deadline, period, id;
    std::chrono::system_clock::time_point beginTime, currentTime;
    //function that does not return anything and takes any amount of args with any type. 
    std::function<void(FuncArgs...)> job;

    bool firstJobReleased = false;
    

  public:

    Task(int offset, int compute_time, int rel_deadline, int period, std::function<void(FuncArgs...)>job, int id, std::chrono::system_clock::time_point beginTime, std::chrono::system_clock::time_point currentTime)
      :offset(offset), compute_time(compute_time), rel_deadline(rel_deadline), period(period), job(job), id(id), beginTime(beginTime), currentTime(currentTime) {}
    

    void launchJob() override{
      if(!firstJobReleased){
        firstJobReleased = true;
      }
      auto& args_tuple = std::any_cast<std::tuple<FuncArgs...>&>(args);
      std::apply(job, args_tuple);
      return;
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
   
};
