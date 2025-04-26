#pragma once

#include "../schedulers/schedulerBase/scheduler.h"
#include "../tasks/task.h"

void runBenchmark(BaseScheduler *scheduler, std::vector<Task>& tasks, int benchmarkDurationSeconds);
