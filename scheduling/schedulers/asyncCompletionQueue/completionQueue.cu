#include "completionQueue.h"

void CompletionQueue::push(CompletionEvent event) {
  {
    std::lock_guard lk(mtx);
    jobCleanUpQueue.push(event);
  }
  cv.notify_one();
}

bool CompletionQueue::pop(CompletionEvent &event) {
  std::unique_lock lk(mtx);
  // Wait until queue is not empty.
  cv.wait(lk, [&] { return !jobCleanUpQueue.empty(); });

  // If queue is not empty
  if (!jobCleanUpQueue.empty()) {
    event = jobCleanUpQueue.front();
    jobCleanUpQueue.pop();
    return true;
  }

  // if queue is empty, return false.
  return false;
}

// need an instance of the queue for the worker thread. Make it static to have
// only one instance.
CompletionQueue &CompletionQueue::getCompletionQueue() {
  static CompletionQueue instance;
  return instance;
}
