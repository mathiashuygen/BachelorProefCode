#include "common/helpFunctions.h"
#include "jobs/busyJob/busyJob.h"
#include "jobs/jobFactory/jobFactory.h"
#include "jobs/matrixMultiplicationJob/matrixMultiplicationJob.h"
#include "jobs/printJob/printJob.h"
#include "jobs/vectorAddJob/vectorAddJob.h"
#include "schedulers/FCFSScheduler/FCFSScheduler.h"
#include "schedulers/JLFPScheduler/JLFP.h"
#include "schedulers/dumbScheduler/dumbScheduler.h"
#include "tasks/task.h"
#include <memory>

int main() {

  std::vector<Task> tasks;

  std::unique_ptr<JobFactoryBase> printJobFactory =
      JobFactory<PrintJob, int, int>::create(20, 10);
  std::unique_ptr<JobFactoryBase> printJobFactory2 =
      JobFactory<PrintJob, int, int>::create(20, 10);
  std::unique_ptr<JobFactoryBase> printJobFactory3 =
      JobFactory<PrintJob, int, int>::create(20, 10);
  std::unique_ptr<JobFactoryBase> printJobFactory4 =
      JobFactory<PrintJob, int, int>::create(20, 10);

  std::unique_ptr<JobFactoryBase> busyJobFactory =
      JobFactory<BusyJob, int, int>::create(10, 10);
  std::unique_ptr<JobFactoryBase> busyJobFactory2 =
      JobFactory<BusyJob, int, int>::create(10, 10);
  std::unique_ptr<JobFactoryBase> vectorAddJobFactory =
      JobFactory<VectorAddJob, int, int>::create(512, 100000);
  std::unique_ptr<JobFactoryBase> vectorAddJobFactory2 =
      JobFactory<VectorAddJob, int, int>::create(512, 100000);
  std::unique_ptr<JobFactoryBase> matrixMulFactory =
      JobFactory<MatrixMultiplicationJob, int, int>::create(512, 4000);

  //  tasks.push_back(Task(10, 10, 60, 5, std::move(busyJobFactory), 2));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(printJobFactory), 1));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(printJobFactory2), 1));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(printJobFactory3), 1));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(printJobFactory4), 1));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(vectorAddJobFactory), 1));
  // tasks.push_back(Task(10, 10, 20, 5, std::move(vectorAddJobFactory2), 1));

  // tasks.push_back(Task(10, 10, 20, 5, std::move(printJobFactory), 1));

  // tasks.push_back(Task(10, 10, 20, 5, std::move(busyJobFactory2), 1));
  tasks.push_back(Task(100, 10000, 2, 1000, std::move(matrixMulFactory), 1));

  JLFP scheduler1(1);
  DumbScheduler scheduler2;
  FCFSScheduler scheduler3;

  while (true) {
    for (Task &task : tasks) {
      if (task.isJobReady()) {
        scheduler1.addJob(task.releaseJob());
      }
    }

    // scheduler1.displayQueueJobs();
    scheduler1.dispatch();
    //    sleep(2000);
  }

  return 0;
}
