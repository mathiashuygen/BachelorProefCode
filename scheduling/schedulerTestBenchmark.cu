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

std::vector<Task> parseTaskSetFromStdin() {
    std::vector<Task> tasks;
    std::vector<std::string> tokens;
    std::string token;

    // Read all input tokens
    while (std::cin >> token) {
        tokens.push_back(token);
    }

    if (tokens.empty()) {
        std::cerr << "No input provided.\n";
        return tasks;
    }

    // First token is number of tasks
    int taskCount = std::stoi(tokens[0]);
    size_t expectedTokens = 1 + taskCount * 7;

    if (tokens.size() < expectedTokens) {
        std::cerr << "Not enough tokens. Expected at least " << expectedTokens
                  << " but got " << tokens.size() << std::endl;
        return tasks;
    }

    size_t idx = 1;
    for (int i = 0; i < taskCount; ++i) {
        std::string type = tokens[idx++];
        int offset = std::stoi(tokens[idx++]);
        int wcet = std::stoi(tokens[idx++]);
        int deadline = std::stoi(tokens[idx++]);
        int period = std::stoi(tokens[idx++]);
        int threadsPerBlock = std::stoi(tokens[idx++]);
        int blockCount = std::stoi(tokens[idx++]);

        std::unique_ptr<JobFactoryBase> factory;

        if (type == "busy") {
            factory = JobFactory<BusyJob, int, int>::create(threadsPerBlock, blockCount);
        } else if (type == "print") {
            factory = JobFactory<PrintJob, int, int>::create(threadsPerBlock, blockCount);
        } else {
            std::cerr << "Unknown job type: " << type << " (task " << i << ")" << std::endl;
            continue;
        }

        tasks.emplace_back(offset, wcet, deadline, period, std::move(factory), i);
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
