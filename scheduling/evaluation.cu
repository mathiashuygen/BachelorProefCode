#include <fstream>
#include <unordered_map>

#include "tasks/task.h"

#include "jobs/busyJob/busyJob.h"
#include "jobs/jobFactory/jobFactory.h"
#include "jobs/matrixMultiplicationJob/matrixMultiplicationJob.h"
#include "jobs/printJob/printJob.h"
#include "jobs/vectorAddJob/vectorAddJob.h"
#include "schedulers/FCFSScheduler/FCFSScheduler.h"
#include "schedulers/JLFPScheduler/JLFP.h"
#include "schedulers/dumbScheduler/dumbScheduler.h"
#include "schedulers/schedulerBase/scheduler.h"

#include "executive/runbenchmark.h"
#include "executive/scheduling.h"

// Parses CLI args of the form --key value into a map
std::unordered_map<std::string, std::string> parseArgs(int argc, char *argv[]) {
  std::unordered_map<std::string, std::string> args;
  for (int i = 1; i < argc - 1; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) == 0) { // starts with "--"
      args[key.substr(2)] = argv[++i];
    }
  }
  return args;
}

std::vector<Task> parseTaskSetFromFile(const std::string &filepath) {
  std::ifstream file(filepath);
  std::vector<std::string> tokens;
  std::string token;

  if (!file) {
    std::cerr << "Failed to open task file: " << filepath << std::endl;
    return {};
  }

  while (file >> token) {
    tokens.push_back(token);
  }

  if (tokens.empty()) {
    std::cerr << "No input in file.\n";
    return {};
  }

  const int taskCount = std::stoi(tokens[0]);
  const size_t expectedTokens = 1 + taskCount * 7;

  if (tokens.size() < expectedTokens) {
    std::cerr << "Not enough tokens in task file. Expected " << expectedTokens
              << ", got " << tokens.size() << std::endl;
    return {};
  }

  std::vector<Task> tasks;
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
      factory =
          JobFactory<BusyJob, int, int>::create(threadsPerBlock, blockCount);
    } else if (type == "print") {
      factory =
          JobFactory<PrintJob, int, int>::create(threadsPerBlock, blockCount);
    } else {
      std::cerr << "Unknown job type: " << type << std::endl;
      return {};
    }

    tasks.emplace_back(offset, wcet, deadline, period, std::move(factory), i);
  }

  return tasks;
}

int main(int argc, char *argv[]) {
  auto args = parseArgs(argc, argv);

  const std::string schedulerType =
      args.count("scheduler") ? args["scheduler"] : "JLFP";
  const std::string taskFilePath =
      args.count("task-file") ? args["task-file"] : "";

  const int tpcSplitDenom = TPC_SPLIT_DENOM;
  // subset when job can't get the amount of TPCs it "requires".
  const float tpcSubset = TPC_SUBSET;

  if (taskFilePath.empty()) {
    std::cerr << "Missing --task-file argument.\n";
    return 1;
  }

  auto scheduler = createScheduler(schedulerType, tpcSplitDenom, tpcSubset);
  if (!scheduler)
    return 1;

  auto tasks = parseTaskSetFromFile(taskFilePath);
  if (tasks.empty()) {
    std::cerr << "Malformed taskset file.\n";
    return 1;
  }

  runBenchmark(scheduler.get(), tasks, 5);

  return 0;
}
