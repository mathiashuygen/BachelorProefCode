
#include "../executive/runbenchmark.h"

// Function to run the benchmark
void runBenchmark(
    BaseScheduler *scheduler,
    std::vector<Task>& tasks,
    int benchmarkDurationSeconds
) {
    // Metrics to track
    int jobsCompleted = 0;
    int jobsMissedDeadline = 0;
    float totalExecutionTime = 0.0f;
    std::vector<float> executionTimes;

    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();

    // Run for a fixed amount of time
    auto endTime = startTime + std::chrono::seconds(benchmarkDurationSeconds);

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
