#!/bin/bash

LIBSMCTRL_PATH="/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl/"

#ls -l $LIBSMCTRL_PATH

nvcc -o main main.cu Tasks/task.cu schedulers/scheduler.cu schedulers/JLFP.cu schedulers/dumbScheduler.cu schedulers/FCFSPartitioning.cu Jobs/kernels/busyKernel.cu \
  Jobs/kernels/printKernel.cu Jobs/job.cu Jobs/printJob.cu Jobs/busyJob.cu \
  common/helpFunctions.cu common/deviceProps.cu common/maskElement.cu \
  -I${LIBSMCTRL_PATH} -lsmctrl -lcuda -L${LIBSMCTRL_PATH}
