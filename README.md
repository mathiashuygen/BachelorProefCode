This repository contains all the code written for my bachelor's thesis.

The directorie "experiments" contain experiments that partition the TPCs on an NVIDIA GPU. 
Running these experiments can be done by executing the shell file in these directories:
```bash
./execute.sh
```
The "workloadManager" directory contains code that lets the user choose which of two types of tasks
they want to launch on the GPU. These two types are:\

- Compute intensive tasks.
- Memory bound tasks. 

Executing the workload manager can again be done by running 
```bash
./execute.sh
```
Both the experiments and the workload manager generate reports that can be analysed in the NVIDIA Nsight Compute application.

