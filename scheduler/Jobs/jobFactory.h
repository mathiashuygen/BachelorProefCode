#ifndef JOB_FACTORY_H
#define JOB_FACTORY_H

#include <memory>
#include <tuple>
#include "job.h"

// Base factory interface
class JobFactoryBase {
public:
    virtual std::unique_ptr<Job> createJob() = 0;
    virtual ~JobFactoryBase() = default;
};

// Templated factory for jobs with FuncArgs and CtorArgs
template<typename JobConcrete, typename... CtorArgs>
class TemplatedJobFactory : public JobFactoryBase {
private:
    std::tuple<CtorArgs...> ctorArgs;

public:
    explicit TemplatedJobFactory(CtorArgs... args) 
        : ctorArgs(std::forward<CtorArgs>(args)...) {}

    std::unique_ptr<Job> createJob() override {
        return std::apply([](auto&&... params) {
            return std::make_unique<JobConcrete>(std::forward<decltype(params)>(params)...);
        }, ctorArgs);
    }
};

// Helper to simplify factory creation
template<template<typename...> class JobType, typename... FuncArgs>
class JobFactory {
public:
    template<typename... CtorArgs>
    static std::unique_ptr<JobFactoryBase> create(CtorArgs&&... args) {
        using ConcreteJob = JobType<FuncArgs...>;
        return std::make_unique<TemplatedJobFactory<ConcreteJob, CtorArgs...>>(
            std::forward<CtorArgs>(args)...
        );
    }
};

#endif
