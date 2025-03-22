/*
 *  Task class that implements the abstract class.
 * */

#include <tuple>
#include <utility>
#include <any>




#ifndef JOB_H
#define JOB_H

class Job{
  public:
    int minimumCores, maximumCores;

    virtual void launchJob() = 0;

    template<typename... Args>
    void execute(Args&&... args){
      this->args = std::make_tuple(std::forward<Args>(args)...);
      launchJob();
      return;
    }


  protected:
    std::any args;

};










#endif // JOB_H
