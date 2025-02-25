FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    nsight-compute-2024.3.0 \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash dockeruser

# Set the working directory
WORKDIR /home/dockeruser/app
# Copy the CUDA source code and the library files
COPY main.cu .
COPY computeIntensive.h .
COPY memoryIntensive.h .
# Change ownership of the files to the non-root user
RUN chown -R dockeruser:dockeruser /home/dockeruser/app

# Switch to the non-root user
USER dockeruser

# Compile the CUDA code and link the library
RUN nvcc -o workload_manager main.cu

# Create a directory for reports
RUN mkdir -p reports

# Run the application with Nsight Compute
CMD ncu --set full \
    --target-processes all \
    -o /home/dockeruser/app/reports/"${REPORT_NAME:-workload_manager_report}" \
    ./workload_manager
