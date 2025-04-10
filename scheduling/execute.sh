#!/bin/bash

LIBSMCTRL_PATH="/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl/"

#ls -l $LIBSMCTRL_PATH

nvcc -o main main.cu Tasks/task.cu schedulers/schedulerBase/scheduler.cu schedulers/JLFPScheduler/JLFP.cu schedulers/dumbScheduler/dumbScheduler.cu \
  schedulers/FCFSScheduler/FCFSScheduler.cu Jobs/kernels/busyKernel.cu \
  Jobs/kernels/printKernel.cu Jobs/jobBase/job.cu Jobs/printJob/printJob.cu Jobs/busyJob/busyJob.cu \
  common/helpFunctions.cu common/deviceProps.cu common/maskElement.cu \
  -I${LIBSMCTRL_PATH} -lsmctrl -lcuda -L${LIBSMCTRL_PATH}
