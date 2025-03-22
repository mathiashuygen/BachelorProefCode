
__global__ static void printMessage(int taskId, int jobId, int loopDuration, float* timing){
  
  float startTime = clock64();
  for(int i = 0; i < loopDuration; i++){
    float y = 0;
    y = sinf(10.2) + cosf(3.1);
  }
  float endTime = clock64();

  *timing = endTime - startTime;

}


