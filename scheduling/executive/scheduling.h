
#include "../schedulers/schedulerBase/scheduler.h"

std::unique_ptr<BaseScheduler> createScheduler(const std::string &type, int tpcSplitDenom);
