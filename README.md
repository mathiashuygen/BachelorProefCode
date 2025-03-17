This repository contains all the code written for my bachelor's thesis.

The directorie "experiments" contain experiments that partition the TPCs on an NVIDIA GPU. 
Running these experiments can be done by executing the shell file in these directories:
```bash
./execute.sh
```
The "workloadManager" directory contains code that lets the user choose which of two types of tasks
they want to launch on the GPU. These two types are:

- Compute intensive tasks.
- Memory bound tasks. 

Executing the workload manager can again be done by running 
```bash
./execute.sh
```
Both the experiments and the workload manager generate reports that can be analysed in the NVIDIA Nsight Compute application.



The "cuda_scheduling_examiner_mirror" comes from the following repository: https://github.com/yalue/cuda_scheduling_examiner_mirror.
I use the repository to visualise the SM usage of kernels. This makes it easy to see the effect that the libsmctrl library 
has on the execution of kernels. My benchmark code is located in the directory src/bachSrcFiles. The code inside here 
is copied from the "busy_kernel" code and adapted such that its execution gives back the relevant information for the python scripts 
to visualise the results. The config file for this benchmark is located at configs/busy_kernel.
Inside the JSON file there are 4 different kernel launch configurations present. Each kernel is launched into a different stream
and each stream has a unique bit mask. The mask can be changed by changing the "sm_mask" value inside the JSON file. 
Running the benchmark can be done by executing the following bash command: 
```bash
./bin/runner configs/busy_kernel 
```

Visualising the SM usage can be done by executing a python script provided by the repository:
```bash
python3 scripts/view_blocksbysm.py 
```

