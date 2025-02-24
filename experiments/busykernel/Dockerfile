FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    nsight-compute-2023.2.0 \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash dockeruser

# Set the working directory
WORKDIR /home/dockeruser/app

RUN git clone http://rtsrv.cs.unc.edu/cgit/cgit.cgi/libsmctrl.git && \
    cd libsmctrl && \
    make NVCC=/usr/local/cuda/bin/nvcc libsmctrl.a

# Copy the CUDA source code and the library files
COPY main.cu .

# Change ownership of the files to the non-root user
RUN chown -R dockeruser:dockeruser /home/dockeruser/app

# Switch to the non-root user
USER dockeruser

# Compile the CUDA code and link the library
RUN nvcc -o busy_kernel main.cu -I/home/dockeruser/app/libsmctrl -L/home/dockeruser/app/libsmctrl -l:libsmctrl.a -lcuda -ldl

# Create a directory for reports
RUN mkdir -p reports

# Run the application with Nsight Compute
CMD ncu --set full \
    --target-processes all \
    --launch-skip 0 \
    --launch-count 1 \
    -o /home/dockeruser/app/reports/"${REPORT_NAME:-busy_kernel_report}" \
    ./busy_kernel
