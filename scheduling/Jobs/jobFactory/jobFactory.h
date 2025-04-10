#ifndef JOB_FACTORY_H
#define JOB_FACTORY_H

#include "../jobBase/job.h"
#include <memory>
#include <utility>

class JobFactoryBase {
public:
  virtual std::unique_ptr<Job> createJob() const = 0;
  virtual ~JobFactoryBase() = default;
};

// Concrete job factory implementation
template <typename JobType, typename... ConstructorArgs>
class JobFactory : public JobFactoryBase {
private:
  // Store constructor arguments as a tuple
  std::tuple<ConstructorArgs...> constructorArgs;

public:
  // Constructor to store arguments
  explicit JobFactory(ConstructorArgs... args)
      : constructorArgs(std::forward<ConstructorArgs>(args)...) {}

  // Override createJob method
  std::unique_ptr<Job> createJob() const override {
    // Use std::apply to unpack the tuple and call the constructor
    return std::apply(
        [](auto &&...args) {
          return std::make_unique<JobType>(
              std::forward<decltype(args)>(args)...);
        },
        constructorArgs);
  }

  // Static method to create a factory
  static std::unique_ptr<JobFactoryBase> create(ConstructorArgs... args) {
    return std::make_unique<JobFactory>(std::forward<ConstructorArgs>(args)...);
  }
};

#endif
