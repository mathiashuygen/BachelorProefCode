cmake_minimum_required(VERSION 3.30)
project(ownCudaPrograms CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(ownCudaPrograms main.cu)

set_target_properties(ownCudaPrograms PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
