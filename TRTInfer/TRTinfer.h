#ifndef TRTINFER_H
#define TRTINFER_H
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>
// #include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <future>
#include <opencv2/opencv.hpp>
#include "utility.h"
#include "config.h"
#include "inference_config.h"
#include "stream_pool.h"
#include "memory_pool.h"
#include "async_infer.h"

class TRTInfer_API Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override;
};

class TRTInfer_API TRTInfer
{
    // init -> load engine -> allocate cuda memory  -> inference
public:
    TRTInfer() = delete;
    /**
     * @param engine_path The weight path of engine
     * @param num_streams Number of CUDA streams for concurrent inference (default: 0, sync mode)
     * @param enable_async Enable async inference (default: false, sync mode)
     * @param use_cvMat Use cv::Mat based async inference (default: true, saves memory if you only use cv::Mat)
     */
    TRTInfer(const std::string &engine_path, int num_streams = 0, bool enable_async = false, bool use_cvMat = true);

    /**
     * @brief Model inference, calling the inner function internally
     * @param input_blob The output blob data consists of the first data being the name of the input tensor, and the second data being the address header of the tensor data
     * @return output blob tensor
     */
    std::unordered_map<std::string, std::shared_ptr<char[]>> operator()(const std::unordered_map<std::string, void *> &input_blob);
    
    
    /**
     * @brief  Model inference based on OpenCV cv::Mat, calling the inner function internally
     * @param input_blob The output blob data consists of the first data being the name of the input tensor, and the second data being the address header of the tensor data
     * @return output blob tensor
     */
    std::unordered_map<std::string, cv::Mat> operator()(const std::unordered_map<std::string, cv::Mat> &input_blob);

    /**
     * @brief Async inference with future return
     * @param input_blob Input data
     * @return Future containing output data
     */
    std::future<std::unordered_map<std::string, std::shared_ptr<char[]>>> 
    infer_async(const std::unordered_map<std::string, void *> &input_blob);
    
    /**
     * @brief Async inference with future return (cv::Mat)
     * @param input_blob Input data
     * @return Future containing output data
     */
    std::future<std::unordered_map<std::string, cv::Mat>> 
    infer_async(const std::unordered_map<std::string, cv::Mat> &input_blob);
    
    /**
     * @brief Async inference with callback
     * @param input_blob Input data
     * @param callback Callback function to be called when inference completes
     */
    void infer_with_callback(const std::unordered_map<std::string, void *> &input_blob,
                             std::function<void(const std::unordered_map<std::string, std::shared_ptr<char[]>>&)> callback);
    
    /**
     * @brief Async inference with callback (cv::Mat)
     * @param input_blob Input data
     * @param callback Callback function to be called when inference completes
     */
    void infer_with_callback(const std::unordered_map<std::string, cv::Mat> &input_blob,
                             std::function<void(const std::unordered_map<std::string, cv::Mat>&)> callback);
    
    /**
     * @brief Wait for all pending async inferences to complete
     */
    void wait_all();
    
    /**
     * @brief Get number of active streams
     */
    int num_streams() const;

    ~TRTInfer();

private:
    void load_engine(const std::string &engine_path);

    void get_InputNames();

    void get_OutputNames();

    void set_OutputBlob();

    std::unordered_map<std::string, std::shared_ptr<char[]>> infer(const std::unordered_map<std::string, void *> &input_blob);

    // for opencv Mat data
    std::unordered_map<std::string, cv::Mat> infer(const std::unordered_map<std::string, cv::Mat> &input_blob);

 private:
    // plugin
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    cudaStream_t stream;
    Logger logger;

    // async components
    bool enable_async_;
    bool use_cvMat_;
    std::shared_ptr<inference::StreamPool> stream_pool_;
    std::shared_ptr<inference::MemoryPool> memory_pool_;
    std::unique_ptr<inference::AsyncInfer<std::unordered_map<std::string, std::shared_ptr<char[]>>>> async_infer_ptr_;
    std::unique_ptr<inference::AsyncInfer<std::unordered_map<std::string, cv::Mat>>> async_infer_mat_;

    // output blob data
    std::unordered_map<std::string, std::shared_ptr<char[]>> output_blob_ptr;

    // data
    std::vector<std::string> input_names, output_names;
    std::unordered_map<std::string, size_t> input_size, output_size;
    std::unordered_map<std::string, std::vector<int>> output_shape;
    cv::Size size;
};

#endif
