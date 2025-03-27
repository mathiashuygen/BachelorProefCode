#include "job.h"

void Job::setMaximumExecutionTime(float time){
  this->releaseTime = time;
}

void Job::setAbsoluteDeadline(float time){
  this->maximalExecutionTime = time;
}

void Job::setReleaseTime(float time){
  this->releaseTime = time;
}

float Job::getAbsoluteDeadline(){
  return this->absoluteDeadline;
}

float Job::getMaximumExecutionTime(){
  return this->maximalExecutionTime;
}

float Job::getReleaseTime(){
  return this->releaseTime;
}

int Job::getMaximumTpcs(){
  return this->maximumTPCs;
}

int Job::getMinimumTPCs(){
  return this->minimumTPCs;
}

void Job::setThreadsPerBlock(int threads){
  this->threadsPerBlock = threads;
}

void Job::setThreadBlocks(int threadBlocks){
  this->threadBlocks = threadBlocks;
}

int Job::getThreadsPerBlock(){
  return this->threadsPerBlock;
}

int Job::getThreadBlocks(){
  return this->threadBlocks;
}
