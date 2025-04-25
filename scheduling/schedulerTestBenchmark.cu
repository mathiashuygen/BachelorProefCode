#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Include scheduler headers
#include "schedulers/FCFSScheduler/FCFSScheduler.h"
#include "schedulers/JLFPScheduler/JLFP.h"
#include "schedulers/dumbScheduler/dumbScheduler.h"

// Include job headers
#include "jobs/busyJob/busyJob.h"
#include "jobs/jobFactory/jobFactory.h"
#include "jobs/printJob/printJob.h"

// Include task header
#include "schedulers/schedulerBase/scheduler.h"
#include "tasks/task.h"

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

// Function to create a scheduler based on type
std::unique_ptr<BaseScheduler> createScheduler(const std::string &type,
                                               int tpcSplitDenom) {
  if (type == "JLFP") {
    return std::make_unique<JLFP>(tpcSplitDenom);
  } else if (type == "FCFS") {
    return std::make_unique<FCFSScheduler>();
  } else if (type == "dumbScheduler") {
    return std::make_unique<DumbScheduler>();
  } else {
    std::cerr << "Unknown scheduler type: " << type << std::endl;
    return nullptr;
  }
}

// Function to run the benchmark
void runBenchmark(
    BaseScheduler *scheduler,
    std::vector<Task>& tasks
) {
  // Metrics to track
  int jobsCompleted = 0;
  int jobsMissedDeadline = 0;
  float totalExecutionTime = 0.0f;
  std::vector<float> executionTimes;

  // Start timing
  auto startTime = std::chrono::high_resolution_clock::now();

  // Run for a fixed amount of time
  const int benchmarkDuration = 15; // seconds
  auto endTime = startTime + std::chrono::seconds(benchmarkDuration);

  while (std::chrono::high_resolution_clock::now() < endTime) {
    // Process any tasks that are ready to release jobs
    for (auto &task : tasks) {
      if (task.isJobReady()) {
        scheduler->addJob(task.releaseJob());
      }
    }

    // Dispatch jobs according to the scheduler
    scheduler->dispatch();
  }

  // Collect and print results
  auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now() - startTime)
                       .count() /
                   1000.0;

  // collected values from scheduler.
  jobsCompleted = scheduler->getJobsCompleted();
  jobsMissedDeadline = scheduler->getDeadlineMisses();

  float throughput = jobsCompleted / totalTime;

  std::cout << "Benchmark Results:" << std::endl;
  std::cout << "Scheduler: " << SCHEDULER_TYPE << std::endl;
  std::cout << "Threads Per Block: " << THREADS_PER_BLOCK << std::endl;
  std::cout << "Block Count: " << BLOCK_COUNT << std::endl;
  std::cout << "Execution time: " << totalTime << " seconds" << std::endl;
  std::cout << "Jobs completed: " << jobsCompleted << std::endl;
  std::cout << "Jobs missed deadline: " << jobsMissedDeadline << std::endl;
  std::cout << "Throughput: " << throughput << " jobs/second" << std::endl;
}

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

  // Create the scheduler based on the compile-time configuration
  auto scheduler = createScheduler(schedulerType, tpcSplitDenom);
  if (!scheduler) {
    return 1;
  }

  auto tasks = get_task_system(numTasks, threadsPerBlock, blockCount);

  // Run the benchmark
  runBenchmark(
      scheduler.get(),
      tasks
  );

  return 0;
}
