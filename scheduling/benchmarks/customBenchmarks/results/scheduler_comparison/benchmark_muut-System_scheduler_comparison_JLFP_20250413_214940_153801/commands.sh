#!/bin/sh
set -e

cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
git rev-parse --abbrev-ref HEAD
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
git rev-parse HEAD
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
uname -a
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
nvcc -o results/scheduler_comparison/JLFP_benchmark -DSCHEDULER_TYPE="JLFP" -DTHREADS_PER_BLOCK=32 -DBLOCK_COUNT=8 -DDATA_SIZE=1024 ../../schedulerTestBenchmark.cu ../../tasks/task.cu ../../schedulers/schedulerBase/scheduler.cu ../../schedulers/JLFPScheduler/JLFP.cu ../../schedulers/dumbScheduler/dumbScheduler.cu ../../schedulers/FCFSScheduler/FCFSScheduler.cu ../../jobs/kernels/busyKernel.cu ../../jobs/kernels/printKernel.cu ../../jobs/jobBase/job.cu ../../jobs/printJob/printJob.cu ../../jobs/busyJob/busyJob.cu ../../common/helpFunctions.cu ../../common/deviceProps.cu ../../common/maskElement.cu -I/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl/ -lsmctrl -lcuda -lcudart -L/home/muut/Documents/github/bachelorProefCode/commonLib/libsmctrl/
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
results/scheduler_comparison/JLFP_benchmark
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
results/scheduler_comparison/JLFP_benchmark
cd /home/muut/Documents/github/bachelorProefCode/scheduling/benchmarks/customBenchmarks
results/scheduler_comparison/JLFP_benchmark
