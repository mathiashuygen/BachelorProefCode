#include "completionQueue.h"
#include <mutex>

void CompletionQueue::push(CompletionEvent event) {
  // lock the queue for the push.
  {
    std::lock_guard lk(mtx);
    // if the shutdown was requested, don't push anything else on the queue.
    if (this->shutdownRequest) {
      return;
    }
    jobCleanUpQueue.push(event);
  }
  // notify the waiting clean up thread that an element has been pushed to the
  // queue => the thread can now pop from the queue.
  cv.notify_one();
}

bool CompletionQueue::pop(CompletionEvent &event) {
  std::unique_lock lk(mtx);
  // Wait until queue is not empty or if there isn't a shutdown request.
  cv.wait(lk,
          [&] { return !jobCleanUpQueue.empty() || this->shutdownRequest; });

  // shutdown requested and there are not items left to clean up.
  if (this->shutdownRequest && jobCleanUpQueue.empty()) {
    return false;
  }

  event = jobCleanUpQueue.front();
  jobCleanUpQueue.pop();
  return true;
}

void CompletionQueue::shutdown() {
  {
    std::lock_guard<std::mutex> lock(this->mtx);
    shutdownRequest = true;
  }
  cv.notify_all();
}

// need an instance of the queue for the worker thread. Make it static to have
// only one instance.
CompletionQueue &CompletionQueue::getCompletionQueue() {
  static CompletionQueue instance;
  return instance;
}
