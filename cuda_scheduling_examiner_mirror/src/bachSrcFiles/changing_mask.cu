#include <codecvt>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <sys/types.h>
#include "../benchmark_gpu_utilities.h"
#include "../library_interface.h"
#include "../benchmark_gpu_utilities.h"
#include "libsmctrl/libsmctrl.h"
#include "../third_party/cJSON.h"




// Holds the local state for one instance of this benchmark.
typedef struct {
  // The CUDA stream with which all operations will be associated.
  cudaStream_t stream;
  int stream_created;
  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times_total;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids_total;

  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times_first_kernel;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids_first_kernel;

 
  // Holds the device copy of the start and end times of each block.
  uint64_t *device_block_times_second_kernel;
  // Holds the device copy of the SMID each block was assigned to.
  uint32_t *device_block_smids_second_kernel; 

  //buffer that stores the result of the loop
  float *device_buffer;
  
  //mask used for the stream.
  uint64_t smMask1;
  uint64_t smMask2;

  int block_count;
  int thread_count;
  // Holds host-side times that are shared with the calling process.
  KernelTimes busy_kernel_times;
  
} TaskState;





// Implements the cleanup function required by the library interface, but is
// also called internally (only during Initialize()) to clean up after errors.
static void Cleanup(void *data) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->busy_kernel_times;
  // Free device memory.
  if (state->device_block_times_total) cudaFree(state->device_block_times_total);
  if (state->device_block_smids_total) cudaFree(state->device_block_smids_total);
  
  if (state->device_block_times_first_kernel) cudaFree(state->device_block_times_first_kernel);
  if (state->device_block_smids_first_kernel) cudaFree(state->device_block_smids_first_kernel);

  if (state->device_block_times_second_kernel) cudaFree(state->device_block_times_second_kernel);
  if (state->device_block_smids_second_kernel) cudaFree(state->device_block_smids_second_kernel);
  



  //free the buffer in which results from the kernel are stored.
  if(state->device_buffer) cudaFree(state->device_buffer);


  // Free host memory.
  if (host_times->block_times) cudaFreeHost(host_times->block_times);
  if (host_times->block_smids) cudaFreeHost(host_times->block_smids);
  if (state->stream_created) {
    // Call CheckCUDAError here to print a message, even though we won't check
    // the return value.
    CheckCUDAError(cudaStreamDestroy(state->stream));
  }



  memset(state, 0, sizeof(*state));
  free(state);
}


// Allocates GPU and CPU memory. Returns 0 on error, 1 otherwise.
static int AllocateMemory(TaskState *state) {
  uint64_t block_times_size = state->block_count * sizeof(uint64_t) * 2;
  uint64_t block_smids_size = state->block_count * sizeof(uint32_t);
  
  uint64_t block_times_size_total = block_times_size * 2;
  uint64_t block_smids_size_total = block_smids_size * 2;


  
  KernelTimes *host_times = &state->busy_kernel_times;

  // Allocate device memory
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times_total),
    block_times_size_total))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids_total),
    block_smids_size_total))) {
    return 0;
  }

  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times_first_kernel),
    block_times_size))) {
    return 0;
  }

  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids_first_kernel),
    block_smids_size))) {
    return 0;
  }


  if (!CheckCUDAError(cudaMalloc(&(state->device_block_times_second_kernel),
    block_times_size))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMalloc(&(state->device_block_smids_second_kernel),
    block_smids_size))) {
    return 0;
  }






  if(!CheckCUDAError(cudaMalloc(&(state->device_buffer), sizeof(float) * 1000000))){
    return 0;
  }
  // Allocate host memory.
  if (!CheckCUDAError(cudaMallocHost(&host_times->block_times,
    block_times_size_total))) {
    return 0;
  }
  if (!CheckCUDAError(cudaMallocHost(&host_times->block_smids,
    block_smids_size_total))) {
    return 0;
  }
  return 1;
}






static int SetMask(TaskState *state){
  
  //create the stream using the stream pointer from the state struct. 
  if(!CheckCUDAError(cudaStreamCreate(&state->stream))){
    std::cout<<"failed to create stream\n";
    return 0;
  };

  //if a mask was provided, set it. the mask will change after the first kernel launch.
  if(state->smMask1){
    libsmctrl_set_stream_mask(state->stream, state->smMask1);
  }
  return 1;
}




static int initKernelConfigs(TaskState *state, char * info){
  cJSON *parsed = NULL;
  cJSON *entry1 = NULL;
  cJSON *entry2 = NULL;
  cJSON *list_entry = NULL;

//parse the config file to get the mask in the additional info section.
//If no mask is given, return from the function without parsing the remaining part of the json file. 
 parsed = cJSON_Parse(info);
  if (!parsed || (parsed->type != cJSON_Array) || !parsed->child) {
    printf("no mask given for kernel launch\n");
    return 1;
  }

  list_entry = parsed->child;
  
  entry1 = cJSON_GetObjectItem(list_entry, "sm_mask1");
  entry2 = cJSON_GetObjectItem(list_entry, "sm_mask2");
  

  if(entry1 && entry2){
    //get the sm mask values and set them.
    uint64_t mask1 = strtoull(entry1->valuestring, NULL, 16);
    uint64_t mask2 = strtoull(entry2->valuestring, NULL, 16);
    std::cout<<"mask 1: "<<mask1<<"\t\n";
    std::cout<<"mask 2: "<<mask2<<"\t\n";

    state->smMask1 = mask1;
    state->smMask2 = mask2;

    
  }
  

  return 1;

}







static void* Initialize(InitializationParameters *params) {
  TaskState *state = NULL;
 // First allocate space for local data.
  state = (TaskState *) calloc(1, sizeof(*state));
  if (!state) return NULL;
  if (!CheckCUDAError(cudaSetDevice(params->cuda_device))) return NULL;
  if (!GetSingleBlockAndGridDimensions(params, &state->thread_count,
    &state->block_count)) {
    Cleanup(state);
    return NULL;
  }
  if (!AllocateMemory(state)) {
    Cleanup(state);
    return NULL;
  }
  if(!initKernelConfigs(state, params->additional_info)){
    Cleanup(state);
    return NULL;
  }
  if (!(SetMask(state))) {
    Cleanup(state);
    return NULL;
  }
  
  state->stream_created = 1;

  return state;
}


// Nothing needs to be copied in for this benchmark.
static int CopyIn(void *data) {
  return 1;
}


// Kernel that performs a simple computation
static __global__ void maxUtilizationKernel(float* output, int n, uint64_t *block_times, uint32_t *block_smids)
{

  uint64_t start_time = GlobalTimer64();
  // First, record the kernel and block start times
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2] = start_time;
    block_smids[blockIdx.x] = GetSMID();
  }
  __syncthreads();  
  
  //actual work of the kernel
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
      // Some computations to keep the SM busy
      float value = 0.0f;
      for (int i = 0; i < n; i++) {
          value += sinf(tid * 0.1f + i) * cosf(tid * 0.1f);
      }
      output[tid] = value;
  }
  // Record the kernel and block end times.
  if (threadIdx.x == 0) {
    block_times[blockIdx.x * 2 + 1] = GlobalTimer64();
  }
}



static __global__ void appendTimesAndSmids(
    uint64_t *total_block_times,
    uint32_t *total_block_smids,
    uint64_t *first_kernel_block_times,
    uint32_t *first_kernel_block_smids,
    uint64_t *second_kernel_block_times, 
    uint32_t *second_kernel_block_smids,
    uint64_t block_times_single_length, 
    uint64_t block_smids_single_length
    ){

  //copying of the block times. 
  for(int i = 0; i < block_times_single_length; i++){
    total_block_times[i] = first_kernel_block_times[i];
    total_block_times[i + block_times_single_length] = second_kernel_block_times[i];
  }

  //copying of the block smids. 
  for(int i = 0; i < block_smids_single_length; i++){
    total_block_smids[i] = first_kernel_block_smids[i];
    total_block_smids[i + block_smids_single_length] = second_kernel_block_smids[i];
  }
}




static int Execute(void *data) {
  TaskState *state = (TaskState *) data;

  //launch the kernel. 
  maxUtilizationKernel<<<state->block_count, state->thread_count, 0, state->stream>>>
    (state->device_buffer, 200000, state->device_block_times_first_kernel, state->device_block_smids_first_kernel);


  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;

  //change the mask of the stream.
  libsmctrl_set_stream_mask(state->stream, state->smMask2);

  //launch the same kernel but this time the stream has a different mask.
  maxUtilizationKernel<<<state->block_count, state->thread_count, 0, state->stream>>>
    (state->device_buffer, 200000, state->device_block_times_second_kernel, state->device_block_smids_second_kernel);


  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;


  //append both arrays by copying the contents into one new total array.
  uint64_t block_times_single_length = state->block_count * 2;
  uint64_t block_smids_single_length = state->block_count;

  //only one thread is needed to calculate the append operation. Could be optimized since the 
  //elements are not data dependent.
  appendTimesAndSmids<<<1, 1, 0, state->stream>>>
    (state->device_block_times_total, state->device_block_smids_total,
     state->device_block_times_first_kernel, state->device_block_smids_first_kernel,
     state->device_block_times_second_kernel, state->device_block_smids_second_kernel,
     block_times_single_length, block_smids_single_length
     );

  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;



  return 1;
}






static int CopyOut(void *data, TimingInformation *times) {
  TaskState *state = (TaskState *) data;
  KernelTimes *host_times = &state->busy_kernel_times;
  uint64_t block_times_count_total = state->block_count * 2 * 2;
  uint64_t block_smids_count_total = state->block_count * 2; 
  memset(times, 0, sizeof(*times));
  


  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_times,
    state->device_block_times_total, block_times_count_total * sizeof(uint64_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }



  if (!CheckCUDAError(cudaMemcpyAsync(host_times->block_smids,
    state->device_block_smids_total, block_smids_count_total * sizeof(uint32_t),
    cudaMemcpyDeviceToHost, state->stream))) {
    return 0;
  }


  if (!CheckCUDAError(cudaStreamSynchronize(state->stream))) return 0;


  host_times->kernel_name = "changing mask";
  host_times->block_count = state->block_count * 2;
  host_times->thread_count = state->thread_count;
  times->kernel_count = 1;
  times->kernel_info = host_times;
  return 1;
}

static const char* GetName(void) {
  return "Mask Change";
}

// This should be the only function we export from the library, to provide
// pointers to all of the other functions.
int RegisterFunctions(BenchmarkLibraryFunctions *functions) {
  functions->initialize = Initialize;
  functions->copy_in = CopyIn;
  functions->execute = Execute;
  functions->copy_out = CopyOut;
  functions->cleanup = Cleanup;
  functions->get_name = GetName;
  return 1;
}

