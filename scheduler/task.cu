#include <iostream>
#include <cuda_runtime.h>
#include <functional>
#include <tuple>
#include <utility>
#include <any>
/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC task.
 *
 */


class TaskBase{
  public:
    virtual void launchJob() = 0;


    template<typename... Args>
    void execute(Args&&... args){
      this->args = std::make_tuple(std::forward<Args>(args)...);
      launchJob();
    }


    virtual ~TaskBase() = default;

  protected:
    std::any args;


};


template<typename... FuncArgs>
class Task: public TaskBase{
  private:
    int offset, compute_time, rel_deadline, period, id;
    
    //function that does not return anything and takes any amount of args with any type. 
    std::function<void(FuncArgs...)> job;
    

  public:

    Task(int offset, int compute_time, int rel_deadline, int period, std::function<void(FuncArgs...)>job, int id)
      :offset(offset), compute_time(compute_time), rel_deadline(rel_deadline), period(period), job(job), id(id) {}
    
    void launchJob() override{
      auto& args_tuple = std::any_cast<std::tuple<FuncArgs...>&>(args);
      std::apply(job, args_tuple);
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
   
};
