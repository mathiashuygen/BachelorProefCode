#include <stdio.h>
#include <stdlib.h>
#include "thrust/device_vector.h"
#include "cuda_runtime.h"
#include "libsmctrl.h"
__host__ __device__ inline float sinsum(float x,int terms)
{
	float x2 = x*x;
	float term = x;   // first term of series
	float sum = term; // sum of terms so far
	for(int n = 1; n < terms; n++){
		term *= -x2 / (2*n*(2*n+1));  // build factorial
		sum += term;
	}
	return sum;
}

__global__ void gpu_sin(float *sums, long steps, long terms,float step_size)
{
	int step = blockIdx.x*blockDim.x+threadIdx.x; // unique thread ID
	if(step<steps){
		float x = step_size*step;
		sums[step] = 0;  // store sin values in array
	}
}

int main(int argc,char *argv[])
{
	long steps = (argc > 1) ? atoi(argv[1]) : 10000000; // get command
	long terms = (argc > 2) ? atoi(argv[2]) : 10000000;     // line arguments
	int threads = 256;
	int blocks = (steps+threads-1)/threads;  // ensure threads*blocks ≥ steps

  cudaStream_t sinusStream;
  cudaStreamCreate(&sinusStream);
  libsmctrl_set_stream_mask((void*)sinusStream, 0x1);

	double step_size = 0.1; // NB n-1 steps between n points

	thrust::device_vector<float> dsums(steps);         // GPU buffer 
	float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer

	gpu_sin<<<blocks,threads, 0, sinusStream>>>(dptr,steps,terms,(float)step_size);
	double gpu_sum = thrust::reduce(dsums.begin(),dsums.end());

	// Trapezoidal Rule Correction
	printf("done computing sinSum\n");
	return 0;
}
