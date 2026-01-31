#ifndef ASYNC_INFER_H
#define ASYNC_INFER_H

#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <future>
#include <opencv2/opencv.hpp>
#include "config.h"
#include "stream_pool.h"
#include "memory_pool.h"
#include "inference_task.h"
#include "inference_config.h"
#include <NvInfer.h>

namespace inference {

template<typename OutputType>
class AsyncInfer {
public:
    using InputType = std::unordered_map<std::string, void*>;
    using InputMatType = std::unordered_map<std::string, cv::Mat>;
    using Callback = std::function<void(const OutputType&)>;
    
    AsyncInfer(nvinfer1::IExecutionContext* context,
               nvinfer1::ICudaEngine* engine,
               std::shared_ptr<StreamPool> stream_pool,
               std::shared_ptr<MemoryPool> memory_pool,
               const std::vector<std::string>& input_names,
               const std::vector<std::string>& output_names,
               const std::unordered_map<std::string, size_t>& input_sizes,
               const std::unordered_map<std::string, size_t>& output_sizes,
               const std::unordered_map<std::string, std::vector<int>>& output_shapes = {});
    
    ~AsyncInfer();
    
    std::future<OutputType> infer_async(const InputType& input_blob);
    std::future<OutputType> infer_async(const InputMatType& input_blob);
    
    void infer_with_callback(const InputType& input_blob, Callback callback);
    void infer_with_callback(const InputMatType& input_blob, Callback callback);
    
    size_t pending_count() const;
    void wait_all();
    void shutdown();
    
 private:
    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine* engine_;
    std::shared_ptr<StreamPool> stream_pool_;
    std::shared_ptr<MemoryPool> memory_pool_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::unordered_map<std::string, size_t> input_sizes_;
    std::unordered_map<std::string, size_t> output_sizes_;
    std::unordered_map<std::string, std::vector<int>> output_shapes_;
    
    std::atomic<bool> running_;
    std::atomic<int> task_counter_;
    
    OutputType infer_impl(const InputType& input_blob, std::shared_ptr<StreamHandle> stream);
    OutputType infer_impl_mat(const InputMatType& input_blob, std::shared_ptr<StreamHandle> stream);
    
    void copy_input_to_device(const InputType& input_blob, std::shared_ptr<StreamHandle> stream);
    void copy_input_to_device(const InputMatType& input_blob, std::shared_ptr<StreamHandle> stream);
    OutputType copy_output_from_device(std::shared_ptr<StreamHandle> stream);
};

template<>
std::unordered_map<std::string, cv::Mat> 
AsyncInfer<std::unordered_map<std::string, cv::Mat>>::copy_output_from_device(
    std::shared_ptr<StreamHandle> stream);

}

#endif
