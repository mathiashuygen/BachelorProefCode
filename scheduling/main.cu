#include "common/helpFunctions.h"
#include "jobs/busyJob/busyJob.h"
#include "jobs/jobFactory/jobFactory.h"
#include "jobs/printJob/printJob.h"
#include "schedulers/FCFSScheduler/FCFSScheduler.h"
#include "schedulers/JLFPScheduler/JLFP.h"
#include "schedulers/dumbScheduler/dumbScheduler.h"
#include "tasks/task.h"
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
        scheduler1.addJob(task.releaseJob());
      }
    }
    // scheduler1.displayQueueJobs();
    scheduler1.dispatch();
    sleep(2000);
  }

  return 0;
}
