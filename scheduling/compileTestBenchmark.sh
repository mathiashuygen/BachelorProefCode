#!/bin/bash

LIBSMCTRL_PATH="/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl/"

nvcc -o benchmark schedulerTestBenchmark.cu tasks/task.cu schedulers/schedulerBase/scheduler.cu schedulers/JLFPScheduler/JLFP.cu schedulers/dumbScheduler/dumbScheduler.cu \
  schedulers/FCFSScheduler/FCFSScheduler.cu jobs/kernels/busyKernel.cu \
  jobs/kernels/printKernel.cu jobs/jobBase/job.cu jobs/printJob/printJob.cu jobs/busyJob/busyJob.cu \
  common/helpFunctions.cu common/deviceProps.cu common/maskElement.cu \
  -I${LIBSMCTRL_PATH} -lsmctrl -lcuda -L${LIBSMCTRL_PATH}
