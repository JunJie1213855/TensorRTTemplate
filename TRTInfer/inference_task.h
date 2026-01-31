#ifndef INFERENCE_TASK_H
#define INFERENCE_TASK_H

#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>
#include <future>
#include "config.h"
#include "stream_pool.h"

namespace inference {

enum class TaskStatus {
    Pending,
    Running,
    Completed,
    Failed
};

template<typename OutputType>
class InferenceTask {
public:
    using OutputFuture = std::future<OutputType>;
    using OutputPromise = std::promise<OutputType>;
    using Callback = std::function<void(const OutputType&)>;
    
    InferenceTask(int task_id, std::shared_ptr<StreamHandle> stream)
        : task_id_(task_id),
          stream_(stream),
          status_(TaskStatus::Pending),
          promise_(std::make_shared<OutputPromise>()) {
    }
    
    ~InferenceTask() = default;
    
    int task_id() const { return task_id_; }
    std::shared_ptr<StreamHandle> stream() const { return stream_; }
    TaskStatus status() const { return status_; }
    
    void set_status(TaskStatus status) {
        status_ = status;
    }
    
    void set_output(const OutputType& output) {
        output_ = output;
        promise_->set_value(output);
        
        if (callback_) {
            callback_(output);
        }
    }
    
    void set_exception(const std::exception& e) {
        promise_->set_exception(std::current_exception());
        status_ = TaskStatus::Failed;
    }
    
    OutputFuture get_future() {
        return promise_->get_future();
    }
    
    void set_callback(Callback cb) {
        callback_ = std::move(cb);
    }
    
    const OutputType& get_output() const {
        return output_;
    }
    
private:
    int task_id_;
    std::shared_ptr<StreamHandle> stream_;
    TaskStatus status_;
    std::shared_ptr<OutputPromise> promise_;
    OutputType output_;
    Callback callback_;
};

}
#endif
