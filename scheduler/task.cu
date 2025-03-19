#include <iostream>
#include <cuda_runtime.h>

/**
 * Representation of a task. Includes all the necessary elements of a PERIODIC task.
 *
 */
class Task{
  private:
    int offset, compute_time, rel_deadline, period, id;
    void (*job) (int, int);
  public:

    Task(int offset, int compute_time, int rel_deadline, int period, void (*job) (int, int), int id){
      this->offset = offset;
      this->compute_time = compute_time;
      this->rel_deadline = rel_deadline;
      this->period = period;
      this->job = job;
      this->id = id;
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

    void launchJob(){
      job<<<1, 1>>>(this->id, this->id);
    }



};
