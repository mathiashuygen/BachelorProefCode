#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>

// Include scheduler headers
#include "schedulers/FCFSScheduler/FCFSScheduler.h"
#include "schedulers/JLFPScheduler/JLFP.h"
#include "schedulers/dumbScheduler/dumbScheduler.h"

// Include job headers
#include "jobs/busyJob/busyJob.h"
#include "jobs/matrixMultiplicationJob/matrixMultiplicationJob.h"
#include "jobs/vectorAddJob/vectorAddJob.h"
#include "jobs/jobFactory/jobFactory.h"
#include "jobs/printJob/printJob.h"

// Include task header
#include "schedulers/schedulerBase/scheduler.h"
#include "tasks/task.h"
#include "executive/scheduling.h"
#include "executive/runbenchmark.h"

#ifndef SCHEDULER_TYPE
#define SCHEDULER_TYPE "JLFP" // Default scheduler type
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 128 // Default threads per block
#endif

#ifndef BLOCK_COUNT
#define BLOCK_COUNT 4 // Default block count
#endif
// example => if a job needs 4 tpcs, the denom will determine if it gets all of
// them or less.
#ifndef TPC_SPLIT_DENOM
#define TPC_SPLIT_DENOM 1
#endif

std::vector<Task> get_task_system(
    int numTasks,
    int threadsPerBlock,
    int blockCount
) {
    std::vector<Task> tasks;

    // Create a mix of job types for the benchmark
    for (int i = 0; i < numTasks; i++) {
        if (i % 2 == 0) {
            // BusyJob tasks
            std::unique_ptr<JobFactoryBase> busyJobFactory =
                    JobFactory<BusyJob, int, int>::create(threadsPerBlock, blockCount);

            const int offset = 0;     // Stagger release times
            const int wcet = 20;
            const int deadline = 10;//;30 * (i + 1); // Stagger deadlines
            const int period = 40;

            tasks.push_back(Task(offset, wcet, deadline, period, std::move(busyJobFactory), i));
        } else {
            // PrintJob tasks
            std::unique_ptr<JobFactoryBase> printJobFactory =
                    JobFactory<PrintJob, int, int>::create(threadsPerBlock, blockCount);

            const int offset = 0;     // Stagger release times
            const int wcet = 20;
            const int deadline = 30 * (i + 1); // Stagger deadlines
            const int period = 40;

            tasks.push_back(Task(offset, wcet, deadline, period, std::move(printJobFactory), i));
        }
    }

    return tasks;
}

int main() {
  const std::string schedulerType = SCHEDULER_TYPE;
  const int threadsPerBlock = THREADS_PER_BLOCK;
  const int blockCount = BLOCK_COUNT;
  const int tpcSplitDenom = TPC_SPLIT_DENOM;
  const int numTasks = 5;

  std::cout << "Starting benchmark with " << schedulerType << " scheduler"
            << std::endl;
  std::cout << "Configuration: " << threadsPerBlock << " threads per block, "
            << std::endl;
  std::cout << "Configuration: " << blockCount << " blocks (count), "
            << std::endl;
  std::cout << "Configuration: " << tpcSplitDenom << " tpc split denominator, "
            << std::endl;
  std::cout << "Configuration: " << numTasks << " tasks, "
            << std::endl;

  auto tasks = get_task_system(numTasks, threadsPerBlock, blockCount);

  // Create the scheduler based on the compile-time configuration
  auto scheduler = createScheduler(schedulerType, tpcSplitDenom);
  if (!scheduler) {
    return 1;
  }

  // Run the benchmark
  runBenchmark(
      scheduler.get(),
      tasks,
      15
  );

  return 0;
}
