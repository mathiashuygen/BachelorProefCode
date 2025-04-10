#include "Jobs/busyJob.h"
#include "Jobs/jobFactory.h"
#include "Jobs/printJob.h"
#include "Tasks/task.h"
#include "common/helpFunctions.h"
#include "schedulers/FCFSPartitioning.h"
#include "schedulers/JLFP.h"
#include "schedulers/dumbScheduler.h"
#include <memory>

int main() {

  std::vector<Task> tasks;

  auto printJobFactory = JobFactory<PrintJob, int, int>::create(200, 1024);
  auto busyJobFactory = JobFactory<BusyJob, int, int>::create(10, 10);

  tasks.push_back(Task(10, 10, 20, 70, std::move(printJobFactory), 1));
  tasks.push_back(Task(10, 10, 60, 100, std::move(busyJobFactory), 2));

  JLFP scheduler1;
  DumbScheduler scheduler2;
  FCFSPartitioning scheduler3;

  while (true) {

    for (Task &task : tasks) {
      if (task.isJobReady()) {
        scheduler2.addJob(task.releaseJob());
      }
    }
    // scheduler1.displayQueueJobs();
    scheduler2.dispatch();
    sleep(2000);
  }

  return 0;
}
